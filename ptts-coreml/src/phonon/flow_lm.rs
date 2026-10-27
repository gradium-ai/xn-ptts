//! Build the flow LM graph: one call of `t` positions against a host-managed ring KV cache.
//!
//! `emb -> transformer layers -> out_norm -> (eos logit, flow_net sample)`, plus each layer's
//! new K and V rows. The same graph at `t = 1` is the decode step and at `t = prefill_len` the
//! batched text prefill.
//!
//! Three things keep it on the Neural Engine, and each one fails silently when broken, by
//! planning the whole graph back onto the CPU:
//! - Every shape is static. One symbolic dimension anywhere disqualifies the graph.
//! - No CoreML `state`: a graph containing one does not compile for the ANE at all. The cache is
//!   ordinary I/O instead, held by the host in IOSurface buffers the ANE reads in place.
//! - Nothing above rank 4 in the attention path, which is why rope is half-split (see `rope`).
//!
//! It is fp16 throughout, which the ANE requires. rope cos/sin and the attention mask arrive
//! ready-made from the host.

use crate::mil::{Builder, DType, Gelu, Var};
use crate::proto::core_ml::specification::Model;
use crate::weights::Weights;

/// The flow LM's sizes, from the checkpoint's config (`ptts::flow_lm::FlowLMConfig`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub d: usize,
    pub heads: usize,
    pub layers: usize,
    pub ff: usize,
    pub ldim: usize,
    pub flow_d: usize,
    pub flow_blocks: usize,
}

impl Dims {
    pub fn hd(&self) -> usize {
        self.d / self.heads
    }
}

const FP: DType = DType::Fp16;

/// A graph and its weights blob.
pub type Built = (Model, Option<Vec<u8>>);

/// Package name of the graph for `t` rows against a `ctx`-slot ring, shared by the exporter and
/// the driver.
pub fn package_name(ctx: usize, t: usize) -> String {
    format!("flow-c{ctx}-t{t}")
}

fn w(b: &mut Builder, wt: &Weights, name: &str) -> Result<Var, String> {
    let (shape, data) = wt.get(name)?;
    Ok(b.weight(data, shape))
}

/// Rope on a half-split layout: the first `HD/2` channels rotate against the second half.
///
/// The checkpoint stores rope in the interleaved convention -- adjacent pairs -- which needs a
/// rank-5 reshape to express, and the Neural Engine tops out at rank 4. `permute_rope_rows`
/// reorders the q and k projection rows once at build time so the projection lands in half-split
/// order instead. Attention is unchanged: the same permutation applied to q and k cancels in
/// `q . k^T`, and v is left alone, so `out_proj` still sees v-space.
///
/// `cos`/`sin` are `[1, 1, t, HD/2]`, one entry per rotating pair.
fn rope(dm: &Dims, b: &mut Builder, x: &Var, cos: &Var, sin: &Var, t: usize) -> Var {
    let (p, hi_, ti) = (dm.hd() / 2, dm.heads as i32, t as i32);
    let lo = b.slice_masked(
        x,
        &[0, 0, 0, 0],
        &[1, hi_, ti, p as i32],
        &[true, true, true, false],
        &[true, true, true, false],
        &[1, dm.heads, t, p],
    );
    let hi = b.slice_masked(
        x,
        &[0, 0, 0, p as i32],
        &[1, hi_, ti, dm.hd() as i32],
        &[true, true, true, false],
        &[true, true, true, false],
        &[1, dm.heads, t, p],
    );
    let lc = b.mul(&lo, cos, &[1, dm.heads, t, p]);
    let hs = b.mul(&hi, sin, &[1, dm.heads, t, p]);
    let out_lo = b.sub(&lc, &hs, &[1, dm.heads, t, p]);
    let ls = b.mul(&lo, sin, &[1, dm.heads, t, p]);
    let hc = b.mul(&hi, cos, &[1, dm.heads, t, p]);
    let out_hi = b.add(&ls, &hc, &[1, dm.heads, t, p]);
    b.concat(&[&out_lo, &out_hi], 3, &[1, dm.heads, t, dm.hd()])
}

/// Reorder the q and k rows of a fused `in_proj` from interleaved to half-split rope order.
///
/// Row `h*HD + 2i` moves to `h*HD + i`, and `h*HD + 2i+1` to `h*HD + HD/2 + i`, per head. The v
/// third is untouched.
fn permute_rope_rows(dm: &Dims, w: &[f32], cols: usize) -> Vec<f32> {
    let half = dm.hd() / 2;
    let mut out = w.to_vec();
    for part in 0..2 {
        // q occupies rows [0, D), k occupies [D, 2D); v stays as it is.
        let base = part * dm.d;
        for h in 0..dm.heads {
            for i in 0..half {
                let src_lo = (base + h * dm.hd() + 2 * i) * cols;
                let src_hi = (base + h * dm.hd() + 2 * i + 1) * cols;
                let dst_lo = (base + h * dm.hd() + i) * cols;
                let dst_hi = (base + h * dm.hd() + half + i) * cols;
                out[dst_lo..dst_lo + cols].copy_from_slice(&w[src_lo..src_lo + cols]);
                out[dst_hi..dst_hi + cols].copy_from_slice(&w[src_hi..src_hi + cols]);
            }
        }
    }
    out
}

/// One transformer layer. `k_all`/`v_all` are the stacked `[L, H, ctx, HD]` ring buffers; this
/// layer's new K and V rows are pushed onto `rows` for the graph's outputs.
#[allow(clippy::too_many_arguments)]
fn layer(
    dm: &Dims,
    b: &mut Builder,
    wt: &Weights,
    i: usize,
    x: &Var,
    (cos, sin, mask): (&Var, &Var, &Var),
    (k_all, v_all): (&Var, &Var),
    rows: &mut Vec<Var>,
    ctx: usize,
    t: usize,
    gelu: Gelu,
) -> Result<Var, String> {
    let p = format!("flow_lm.transformer.layers.{i}");
    let (hi_, di) = (dm.heads as i32, dm.hd() as i32);

    let n1w = w(b, wt, &format!("{p}.norm1.weight"))?;
    let n1b = w(b, wt, &format!("{p}.norm1.bias"))?;
    let h = b.layer_norm(x, &n1w, &n1b, 1e-5);
    let inw = {
        let (shape, data) = wt.get(&format!("{p}.self_attn.in_proj.weight"))?;
        b.weight(&permute_rope_rows(dm, data, shape[1]), shape)
    };
    let qkv = b.linear(&h, &inw, None, &[1, t, 3 * dm.d]);
    let qkv = b.reshape(&qkv, &[1, t, 3, dm.heads, dm.hd()]);
    let qkv = b.transpose(&qkv, &[2, 0, 3, 1, 4], &[3, 1, dm.heads, t, dm.hd()]);
    let part = |b: &mut Builder, k: i32| {
        let s = b.slice_masked(
            &qkv,
            &[k, 0, 0, 0, 0],
            &[k + 1, 1, hi_, 0, di],
            &[false, true, true, true, true],
            &[false, true, true, true, true],
            &[1, 1, dm.heads, t, dm.hd()],
        );
        b.reshape(&s, &[1, dm.heads, t, dm.hd()])
    };
    let q = part(b, 0);
    let k = part(b, 1);
    let v = part(b, 2);
    let q = rope(dm, b, &q, cos, sin, t);
    let k = rope(dm, b, &k, cos, sin, t);

    // One stacked buffer for all layers rather than one per layer: binding 48 buffers a call
    // cost 0.31 ms on the M5 ANE against 4, and the phone showed the same ratio.
    let (i0, i1) = (i as i32, i as i32 + 1);
    let sl = |b: &mut Builder, t: &Var| {
        b.slice_masked(
            t,
            &[i0, 0, 0, 0],
            &[i1, 0, 0, 0],
            &[false, true, true, true],
            &[false, true, true, true],
            &[1, dm.heads, ctx, dm.hd()],
        )
    };
    let (kc, vc) = (sl(b, k_all), sl(b, v_all));
    // Attention runs over the ring plus this call's rows, and the graph emits only the rows: the
    // host writes them into the ring, so the 15 MB cache is never shifted or written back.
    let aw = ctx + t;
    let katt = b.concat(&[&kc, &k], 2, &[1, dm.heads, aw, dm.hd()]);
    let vatt = b.concat(&[&vc, &v], 2, &[1, dm.heads, aw, dm.hd()]);
    rows.push(k);
    rows.push(v);

    let att = b.matmul(&q, &katt, true, &[1, dm.heads, t, aw]);
    let sc = b.scalar_f32((dm.hd() as f32).powf(-0.5));
    let att = b.mul(&att, &sc, &[1, dm.heads, t, aw]);
    let att = b.add(&att, mask, &[1, dm.heads, t, aw]);
    let att = b.softmax(&att, -1);
    let o = b.matmul(&att, &vatt, false, &[1, dm.heads, t, dm.hd()]);
    let o = b.transpose(&o, &[0, 2, 1, 3], &[1, t, dm.heads, dm.hd()]);
    let o = b.reshape(&o, &[1, t, dm.d]);
    let ow = w(b, wt, &format!("{p}.self_attn.out_proj.weight"))?;
    let o = b.linear(&o, &ow, None, &[1, t, dm.d]);
    let x = b.add(x, &o, &[1, t, dm.d]);

    let n2w = w(b, wt, &format!("{p}.norm2.weight"))?;
    let n2b = w(b, wt, &format!("{p}.norm2.bias"))?;
    let h = b.layer_norm(&x, &n2w, &n2b, 1e-5);
    let l1 = w(b, wt, &format!("{p}.linear1.weight"))?;
    let l2 = w(b, wt, &format!("{p}.linear2.weight"))?;
    let h = b.linear(&h, &l1, None, &[1, t, dm.ff]);
    let h = b.gelu(&h, gelu);
    let h = b.linear(&h, &l2, None, &[1, t, dm.d]);
    Ok(b.add(&x, &h, &[1, t, dm.d]))
}

/// The norm closing each time embedder, as the checkpoint was trained with
/// (`ptts::flow_lm::TimeRmsNorm`, `flow_lm.time_rms_norm` in its config).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimeNorm {
    /// `h * a / sqrt(mean(h^2) + eps)`, uncentred, as audiocraft computes it.
    Rms,
    /// `h * a / sqrt(var(h) + eps)`, the variance centred and unbiased, as pocket-tts computes
    /// it. The mean is still not subtracted from `h`.
    #[default]
    RmsVar,
}

/// `TimestepEmbedder` at a fixed time, in f32 on the host: the step always goes from s = 0 to
/// t = 1, so the embedding is a constant of the checkpoint and the graph only needs its value.
/// It ends in `norm`, which has to be the one the checkpoint was trained with: the two only
/// agree when the mean is near zero, true enough for one checkpoint to look exact and wrong for
/// another.
fn time_embed(wt: &Weights, idx: usize, t: f32, norm: TimeNorm) -> Result<Vec<f32>, String> {
    let p = format!("flow_lm.flow_net.time_embed.{idx}");
    let freqs = wt.data(&format!("{p}.freqs"))?;
    let mut x: Vec<f32> = freqs.iter().map(|f| (t * f).cos()).collect();
    x.extend(freqs.iter().map(|f| (t * f).sin()));
    let linear = |x: &[f32], name: &str| -> Result<Vec<f32>, String> {
        let (shape, w) = wt.get(&format!("{p}.{name}.weight"))?;
        let b = wt.data(&format!("{p}.{name}.bias"))?;
        let cols = shape[1];
        Ok((0..shape[0])
            .map(|o| {
                b[o] + w[o * cols..(o + 1) * cols].iter().zip(x).map(|(a, b)| a * b).sum::<f32>()
            })
            .collect())
    };
    let h: Vec<f32> = linear(&x, "mlp.0")?.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    let h = linear(&h, "mlp.2")?;
    let n = h.len() as f32;
    let ms = match norm {
        TimeNorm::Rms => h.iter().map(|v| v * v).sum::<f32>() / n,
        TimeNorm::RmsVar => {
            let mean = h.iter().sum::<f32>() / n;
            h.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (n - 1.0)
        }
    };
    let inv = 1.0 / (ms + 1e-5).sqrt();
    let alpha = wt.data(&format!("{p}.mlp.3.alpha"))?;
    Ok(h.iter().zip(alpha).map(|(v, a)| v * inv * a).collect())
}

fn modulate(b: &mut Builder, x: &Var, shift: &Var, scale: &Var, n: usize) -> Var {
    let one = b.scalar_f32(1.0);
    let s = b.add(scale, &one, &[1, n]);
    let x = b.mul(x, &s, &[1, n]);
    b.add(&x, shift, &[1, n])
}

fn flow_net(
    dm: &Dims,
    b: &mut Builder,
    wt: &Weights,
    cond: &Var,
    noise: &Var,
    time_norm: TimeNorm,
) -> Result<Var, String> {
    let p = "flow_lm.flow_net";
    let iw = w(b, wt, &format!("{p}.input_proj.weight"))?;
    let ib = w(b, wt, &format!("{p}.input_proj.bias"))?;
    let mut x = b.linear(noise, &iw, Some(&ib), &[1, dm.flow_d]);
    // The two time conditions averaged, as `SimpleMLPAdaLN::forward` does.
    let (t0, t1) = (time_embed(wt, 0, 0.0, time_norm)?, time_embed(wt, 1, 1.0, time_norm)?);
    let ts: Vec<f32> = t0.iter().zip(&t1).map(|(a, b)| (a + b) * 0.5).collect();
    let ts = b.const_f32(&ts, &[1, dm.flow_d]);
    let cw = w(b, wt, &format!("{p}.cond_embed.weight"))?;
    let cb = w(b, wt, &format!("{p}.cond_embed.bias"))?;
    let c = b.linear(cond, &cw, Some(&cb), &[1, dm.flow_d]);
    let y = b.add(&ts, &c, &[1, dm.flow_d]);
    let fd = dm.flow_d as i32;

    for i in 0..dm.flow_blocks {
        let q = format!("{p}.res_blocks.{i}");
        let aw = w(b, wt, &format!("{q}.adaLN_modulation.1.weight"))?;
        let ab = w(b, wt, &format!("{q}.adaLN_modulation.1.bias"))?;
        let ys = b.silu(&y);
        let ada = b.linear(&ys, &aw, Some(&ab), &[1, 3 * dm.flow_d]);
        let shift = b.slice(&ada, &[0, 0], &[1, fd], &[1, dm.flow_d]);
        let scale = b.slice(&ada, &[0, fd], &[1, 2 * fd], &[1, dm.flow_d]);
        let gate = b.slice(&ada, &[0, 2 * fd], &[1, 3 * fd], &[1, dm.flow_d]);
        let lw = w(b, wt, &format!("{q}.in_ln.weight"))?;
        let lb = w(b, wt, &format!("{q}.in_ln.bias"))?;
        let h = b.layer_norm(&x, &lw, &lb, 1e-6);
        let h = modulate(b, &h, &shift, &scale, dm.flow_d);
        let m0 = w(b, wt, &format!("{q}.mlp.0.weight"))?;
        let m0b = w(b, wt, &format!("{q}.mlp.0.bias"))?;
        let h = b.linear(&h, &m0, Some(&m0b), &[1, dm.flow_d]);
        let h = b.silu(&h);
        let m2 = w(b, wt, &format!("{q}.mlp.2.weight"))?;
        let m2b = w(b, wt, &format!("{q}.mlp.2.bias"))?;
        let h = b.linear(&h, &m2, Some(&m2b), &[1, dm.flow_d]);
        let h = b.mul(&gate, &h, &[1, dm.flow_d]);
        x = b.add(&x, &h, &[1, dm.flow_d]);
    }

    let q = format!("{p}.final_layer");
    let aw = w(b, wt, &format!("{q}.adaLN_modulation.1.weight"))?;
    let ab = w(b, wt, &format!("{q}.adaLN_modulation.1.bias"))?;
    let ys = b.silu(&y);
    let ada = b.linear(&ys, &aw, Some(&ab), &[1, 2 * dm.flow_d]);
    let shift = b.slice(&ada, &[0, 0], &[1, fd], &[1, dm.flow_d]);
    let scale = b.slice(&ada, &[0, fd], &[1, 2 * fd], &[1, dm.flow_d]);
    let ones = b.const_f32(&vec![1f32; dm.flow_d], &[dm.flow_d]);
    let zeros = b.const_f32(&vec![0f32; dm.flow_d], &[dm.flow_d]);
    let h = b.layer_norm(&x, &ones, &zeros, 1e-6);
    let h = modulate(b, &h, &shift, &scale, dm.flow_d);
    let fw = w(b, wt, &format!("{q}.linear.weight"))?;
    let fb = w(b, wt, &format!("{q}.linear.bias"))?;
    Ok(b.linear(&h, &fw, Some(&fb), &[1, dm.ldim]))
}

/// The flow LM for `t` positions against a `ctx`-slot ring cache.
///
/// Inputs: `emb [1, t, D]` (the host applies `input_linear` and the speaker embedding, so the
/// same graph serves prefill and decode), `cos`/`sin [1, 1, t, HD/2]`, `mask [1, 1, t, ctx + t]`,
/// `noise [1, LDIM]`, and the ring `k_all_in`/`v_all_in [L, H, ctx, HD]`. Outputs: `next_latent`
/// and `eos` for the last position, and `k_new`/`v_new [L, H, t, HD]` for the host to store.
/// `time_norm` is the checkpoint's time-embedding norm, baked into the graph as a constant, and
/// `gelu` its FFN activation.
pub fn build(
    wt: &Weights,
    dm: &Dims,
    ctx: usize,
    t: usize,
    time_norm: TimeNorm,
    gelu: Gelu,
) -> Result<Built, String> {
    let mut b = Builder::new().with_float(FP);
    let emb = b.input("emb", FP, &[1, t, dm.d]);
    let cos = b.input("cos", FP, &[1, 1, t, dm.hd() / 2]);
    let sin = b.input("sin", FP, &[1, 1, t, dm.hd() / 2]);
    let mask = b.input("mask", FP, &[1, 1, t, ctx + t]);
    let noise = b.input("noise", FP, &[1, dm.ldim]);
    let k_all = b.input("k_all_in", FP, &[dm.layers, dm.heads, ctx, dm.hd()]);
    let v_all = b.input("v_all_in", FP, &[dm.layers, dm.heads, ctx, dm.hd()]);

    let mut x = emb;
    let mut rows: Vec<Var> = Vec::new();
    for i in 0..dm.layers {
        let (rope, kv) = ((&cos, &sin, &mask), (&k_all, &v_all));
        x = layer(dm, &mut b, wt, i, &x, rope, kv, &mut rows, ctx, t, gelu)?;
    }
    let onw = w(&mut b, wt, "flow_lm.out_norm.weight")?;
    let onb = w(&mut b, wt, "flow_lm.out_norm.bias")?;
    let h = b.layer_norm(&x, &onw, &onb, 1e-5);
    // Only the final position feeds the heads.
    let last = b.slice_tail(&h, 1, 1, &[1, 1, dm.d]);
    let t_out = b.reshape(&last, &[1, dm.d]);

    let ew = w(&mut b, wt, "flow_lm.out_eos.weight")?;
    let eb = w(&mut b, wt, "flow_lm.out_eos.bias")?;
    let eos = b.linear(&t_out, &ew, Some(&eb), &[1, 1]);
    let dir = flow_net(dm, &mut b, wt, &t_out, &noise, time_norm)?;
    let next = b.add(&noise, &dir, &[1, dm.ldim]);
    let next = b.alias(&next, "next_latent");
    let eos = b.alias(&eos, "eos");
    // Only this call's rows, for every layer: [L, H, t, HD], 18 KB at t = 1.
    let ks: Vec<&Var> = rows.iter().step_by(2).collect();
    let vs: Vec<&Var> = rows.iter().skip(1).step_by(2).collect();
    let k = b.concat(&ks, 0, &[dm.layers, dm.heads, t, dm.hd()]);
    let v = b.concat(&vs, 0, &[dm.layers, dm.heads, t, dm.hd()]);
    let k_new = b.alias(&k, "k_new");
    let v_new = b.alias(&v, "v_new");
    Ok(b.finish_with_weights(&[&next, &eos, &k_new, &v_new]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_time_norms_follow_their_formulas() {
        // Zero weights leave `mlp.2`'s bias as the row reaching the norm: one with a clear mean,
        // where centring matters, and alpha = 2.
        let xs = [1.0f32, 2.0, 3.0, 6.0];
        let p = "flow_lm.flow_net.time_embed.0";
        let t = |name: &str, shape: &[usize], data: Vec<f32>| {
            (format!("{p}.{name}"), (shape.to_vec(), data))
        };
        let wt = Weights::from_tensors(
            [
                t("freqs", &[1], vec![0.0]),
                t("mlp.0.weight", &[4, 2], vec![0.0; 8]),
                t("mlp.0.bias", &[4], vec![0.0; 4]),
                t("mlp.2.weight", &[4, 4], vec![0.0; 16]),
                t("mlp.2.bias", &[4], xs.to_vec()),
                t("mlp.3.alpha", &[4], vec![2.0; 4]),
            ]
            .into(),
        );
        let (n, eps) = (4.0f32, 1e-5f32);
        let mean = xs.iter().sum::<f32>() / n;
        let var = xs.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (n - 1.0);
        let ms = xs.iter().map(|v| v * v).sum::<f32>() / n;
        for (norm, den) in [(TimeNorm::Rms, ms), (TimeNorm::RmsVar, var)] {
            let got = time_embed(&wt, 0, 1.0, norm).unwrap();
            for (g, v) in got.iter().zip(xs) {
                assert!((g - 2.0 * v / (den + eps).sqrt()).abs() < 1e-5, "{norm:?}: {got:?}");
            }
        }
    }
}
