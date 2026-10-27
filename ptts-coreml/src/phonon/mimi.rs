//! Build the Mimi decoder graph.
//!
//! One Mimi frame: `latent -> output_proj -> depthwise upsample -> transformer -> SEANet
//! (one transposed conv per ratio) -> PCM`, sized from the checkpoint's config.
//!
//! Streaming state travels as ordinary inputs and outputs rather than CoreML state, because
//! CoreML states are fp16-only and Mimi has to run in f32. Almost all of it is the transformer
//! KV cache.

use crate::mil::{Builder, DType, Gelu, Var};
use crate::weights::Weights;

/// The Mimi decoder's sizes, from the checkpoint's config (`ptts::mimi::MimiConfig`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dims {
    /// Latent width, the flow LM's `ldim`.
    pub ldim: usize,
    /// Width of the upsampled latents the transformer reads (`dimension`).
    pub dim: usize,
    /// The transformer: width, heads, layers and FFN width.
    pub d_model: usize,
    pub heads: usize,
    pub layers: usize,
    pub ff: usize,
    /// Width of the transformer's output and of the SEANet decoder's input
    /// (`decoder_dimension`, else `dimension`).
    pub out_dim: usize,
    /// The SEANet decoder.
    pub n_filters: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub compress: usize,
    /// Transformer positions per latent frame, the upsampling stride.
    pub steps: usize,
}

impl Default for Dims {
    /// The decoder every bundle had before `bundle.json` described it.
    fn default() -> Self {
        Dims {
            ldim: 32,
            dim: 512,
            d_model: 512,
            heads: 8,
            layers: 2,
            ff: 2048,
            out_dim: 512,
            n_filters: 64,
            ratios: vec![6, 5, 4],
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            compress: 2,
            steps: 16,
        }
    }
}

impl Dims {
    pub fn hd(&self) -> usize {
        self.d_model / self.heads
    }

    /// PCM samples per latent frame.
    pub fn frame(&self) -> usize {
        self.steps * self.ratios.iter().product::<usize>()
    }

    /// Slots of the transformer KV cache for an attention window of `window` positions.
    ///
    /// The reference keeps `window - 1` positions between calls and appends the call's `steps`
    /// new rows before attending, so every query sees `window` positions, itself included. This
    /// cache keeps one more, which the mask never shows, so bundles exported with it still load.
    /// A cache of only `window` slots, new rows included, would drop up to `steps` of the oldest
    /// ones.
    pub fn cache_len(&self, window: usize) -> usize {
        window + self.steps
    }

    /// Package name of the Mimi graph for a window, so a cached graph of another cache length is
    /// never picked up in its place.
    pub fn package_name(&self, window: usize) -> String {
        format!("mimi-c{}", self.cache_len(window))
    }

    /// Every streaming buffer, in the order the graph takes and returns them. `ctx` is the cache
    /// length, `cache_len(window)`. The convolutions' buffers are named after their index in the
    /// SEANet decoder.
    pub fn state_spec(&self, ctx: usize) -> Vec<(String, Vec<usize>)> {
        let mut v: Vec<(String, Vec<usize>)> = vec![
            ("up_partial".into(), vec![1, self.dim, self.steps]),
            ("c0_prev".into(), vec![1, self.out_dim, self.kernel_size - 1]),
        ];
        let mut ch = self.n_filters << self.ratios.len();
        for (i, &r) in self.ratios.iter().enumerate() {
            ch /= 2;
            v.push((format!("t{}_partial", 3 * i + 2), vec![1, ch, r]));
            v.push((format!("r{}_prev", 3 * i + 3), vec![1, ch, self.residual_kernel_size - 1]));
        }
        let last = 3 * self.ratios.len() + 2;
        v.push((format!("c{last}_prev"), vec![1, self.n_filters, self.last_kernel_size - 1]));
        for i in 0..self.layers {
            v.push((format!("k{i}"), vec![1, self.heads, ctx, self.hd()]));
            v.push((format!("v{i}"), vec![1, self.heads, ctx, self.hd()]));
        }
        v
    }

    /// The sizes as `bundle.json` stores them.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "ldim": self.ldim, "dim": self.dim, "d_model": self.d_model, "heads": self.heads,
            "layers": self.layers, "ff": self.ff, "out_dim": self.out_dim,
            "n_filters": self.n_filters, "ratios": self.ratios, "kernel_size": self.kernel_size,
            "residual_kernel_size": self.residual_kernel_size,
            "last_kernel_size": self.last_kernel_size, "compress": self.compress,
            "steps": self.steps,
        })
    }

    /// The sizes from `bundle.json`; [`Dims::default`] for a bundle written before it held them.
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        if v.is_null() {
            return Ok(Self::default());
        }
        let n = |k: &str| {
            v[k].as_u64().map(|n| n as usize).ok_or(format!("bundle.json has no mimi.{k}"))
        };
        let ratios = v["ratios"].as_array().ok_or("bundle.json has no mimi.ratios")?;
        Ok(Dims {
            ldim: n("ldim")?,
            dim: n("dim")?,
            d_model: n("d_model")?,
            heads: n("heads")?,
            layers: n("layers")?,
            ff: n("ff")?,
            out_dim: n("out_dim")?,
            n_filters: n("n_filters")?,
            ratios: ratios
                .iter()
                .map(|r| r.as_u64().map(|r| r as usize).ok_or("bad mimi.ratios"))
                .collect::<Result<_, _>>()?,
            kernel_size: n("kernel_size")?,
            residual_kernel_size: n("residual_kernel_size")?,
            last_kernel_size: n("last_kernel_size")?,
            compress: n("compress")?,
            steps: n("steps")?,
        })
    }
}

fn w(b: &mut Builder, wt: &Weights, name: &str) -> Result<Var, String> {
    let (shape, data) = wt.get(name)?;
    // Mimi stays f32 -- no quantisation -- but its weights still belong in the blob.
    Ok(b.weight(data, shape))
}

/// Streaming conv: prepend the carried left context, convolve, return the new tail.
#[allow(clippy::too_many_arguments)]
fn sconv(
    b: &mut Builder,
    x: &Var,
    buf: &Var,
    wv: &Var,
    bv: &Var,
    cin: usize,
    cout: usize,
    tin: usize,
    k: usize,
    tp: usize,
) -> (Var, Var) {
    let full = tp + tin;
    let cat = b.concat(&[buf, x], 2, &[1, cin, full]);
    let out = b.conv1d(&cat, wv, Some(bv), 1, &[1, cout, full - k + 1]);
    let tail =
        b.slice(&cat, &[0, 0, (full - tp) as i32], &[1, cin as i32, full as i32], &[1, cin, tp]);
    (out, tail)
}

/// Streaming transposed conv: overlap-add the carried tail into the head of this output.
#[allow(clippy::too_many_arguments)]
fn sconvtr(
    b: &mut Builder,
    x: &Var,
    buf: &Var,
    wv: &Var,
    bv: Option<&Var>,
    cout: usize,
    tin: usize,
    stride: usize,
    k: usize,
    groups: usize,
) -> (Var, Var) {
    let full = (tin - 1) * stride + k;
    let pt = k - stride;
    let y = b.conv_transpose1d(x, wv, bv, stride, groups, &[1, cout, full]);
    let head = b.slice(&y, &[0, 0, 0], &[1, cout as i32, pt as i32], &[1, cout, pt]);
    let head = b.add(&head, buf, &[1, cout, pt]);
    let rest =
        b.slice(&y, &[0, 0, pt as i32], &[1, cout as i32, full as i32], &[1, cout, full - pt]);
    let joined = b.concat(&[&head, &rest], 2, &[1, cout, full]);
    let out =
        b.slice(&joined, &[0, 0, 0], &[1, cout as i32, (full - pt) as i32], &[1, cout, full - pt]);
    let mut tail = b.slice(
        &joined,
        &[0, 0, (full - pt) as i32],
        &[1, cout as i32, full as i32],
        &[1, cout, pt],
    );
    if let Some(bias) = bv {
        // ptts stores the tail with the bias removed, so it is not added twice.
        let br = b.reshape(bias, &[1, cout, 1]);
        tail = b.sub(&tail, &br, &[1, cout, pt]);
    }
    (out, tail)
}

#[allow(clippy::too_many_arguments)]
fn resblock(
    b: &mut Builder,
    wt: &Weights,
    idx: usize,
    x: &Var,
    prev: &Var,
    cin: usize,
    mid: usize,
    t: usize,
    k: usize,
) -> Result<(Var, Var), String> {
    let p = format!("mimi.decoder.model.{idx}");
    let w1 = w(b, wt, &format!("{p}.block.1.conv.weight"))?;
    let b1 = w(b, wt, &format!("{p}.block.1.conv.bias"))?;
    let w3 = w(b, wt, &format!("{p}.block.3.conv.weight"))?;
    let b3 = w(b, wt, &format!("{p}.block.3.conv.bias"))?;
    let h = b.elu(x);
    let (h, tail) = sconv(b, &h, prev, &w1, &b1, cin, mid, t, k, k - 1);
    let h = b.elu(&h);
    let h = b.conv1d(&h, &w3, Some(&b3), 1, &[1, cin, t]);
    Ok((b.add(x, &h, &[1, cin, t]), tail))
}

/// The Mimi graph for one frame of a `dm`-sized decoder, with a `ctx`-slot transformer cache
/// (`dm.cache_len(window)`) and `gelu` as its transformer's FFN activation.
///
/// f32 throughout, so it runs on the CPU: the vocoder is kept at its original precision, and
/// the Neural Engine is fp16-only.
pub fn build(
    wt: &Weights,
    dm: &Dims,
    ctx: usize,
    gelu: Gelu,
) -> Result<super::flow_lm::Built, String> {
    let fp = DType::Fp32;
    let mut b = Builder::new().with_float(fp);
    let (steps, heads, hd, d) = (dm.steps, dm.heads, dm.hd(), dm.d_model);
    let latent = b.input("latent", fp, &[1, 1, dm.ldim]);
    let cos = b.input("cos", fp, &[1, 1, steps, hd]);
    let sin = b.input("sin", fp, &[1, 1, steps, hd]);
    let mask = b.input("mask", fp, &[1, 1, steps, ctx]);
    let spec = dm.state_spec(ctx);
    let st: Vec<Var> = spec.iter().map(|(n, sh)| b.input(n, fp, sh)).collect();
    // The convolutions' buffers, then each layer's K and V.
    let (conv_st, kv_st) = st.split_at(spec.len() - 2 * dm.layers);

    // denormalise, then the latent width to the transformer's input width
    let mean = w(&mut b, wt, "flow_lm.emb_mean")?;
    let sd = w(&mut b, wt, "flow_lm.emb_std")?;
    let x = b.mul(&latent, &sd, &[1, 1, dm.ldim]);
    let x = b.add(&x, &mean, &[1, 1, dm.ldim]);
    let x = b.transpose(&x, &[0, 2, 1], &[1, dm.ldim, 1]);
    let opw = w(&mut b, wt, "mimi.quantizer.output_proj.weight")?;
    let x = b.conv1d(&x, &opw, None, 1, &[1, dm.dim, 1]);

    // depthwise upsample from the latent frame rate to the transformer's
    let upw = w(&mut b, wt, "mimi.upsample.convtr.convtr.weight")?;
    let (x, up_o) =
        sconvtr(&mut b, &x, &conv_st[0], &upw, None, dm.dim, 1, steps, 2 * steps, dm.dim);

    // transformer over the new positions, behind projections where the widths differ
    let mut h = b.transpose(&x, &[0, 2, 1], &[1, steps, dm.dim]);
    if d != dm.dim {
        let ipw = w(&mut b, wt, "mimi.decoder_transformer.input_proj.weight")?;
        h = b.linear(&h, &ipw, None, &[1, steps, d]);
    }
    let mut kv_out = Vec::new();
    for i in 0..dm.layers {
        let p = format!("mimi.decoder_transformer.transformer.layers.{i}");
        let n1w = w(&mut b, wt, &format!("{p}.norm1.weight"))?;
        let n1b = w(&mut b, wt, &format!("{p}.norm1.bias"))?;
        let n = b.layer_norm(&h, &n1w, &n1b, 1e-5);
        let inw = w(&mut b, wt, &format!("{p}.self_attn.in_proj.weight"))?;
        let qkv = b.linear(&n, &inw, None, &[1, steps, 3 * d]);
        let qkv = b.reshape(&qkv, &[1, steps, 3, heads, hd]);
        let qkv = b.transpose(&qkv, &[2, 0, 3, 1, 4], &[3, 1, heads, steps, hd]);
        let part = |b: &mut Builder, k: i32| {
            let s = b.slice(
                &qkv,
                &[k, 0, 0, 0, 0],
                &[k + 1, 1, heads as i32, steps as i32, hd as i32],
                &[1, 1, heads, steps, hd],
            );
            b.reshape(&s, &[1, heads, steps, hd])
        };
        let q = part(&mut b, 0);
        let k = part(&mut b, 1);
        let v = part(&mut b, 2);
        let q = rope(&mut b, &q, &cos, &sin, heads, steps, hd);
        let k = rope(&mut b, &k, &cos, &sin, heads, steps, hd);

        let kin = &kv_st[2 * i];
        let vin = &kv_st[2 * i + 1];
        let kk = b.slice(
            kin,
            &[0, 0, steps as i32, 0],
            &[1, heads as i32, ctx as i32, hd as i32],
            &[1, heads, ctx - steps, hd],
        );
        let vv = b.slice(
            vin,
            &[0, 0, steps as i32, 0],
            &[1, heads as i32, ctx as i32, hd as i32],
            &[1, heads, ctx - steps, hd],
        );
        let knew = b.concat(&[&kk, &k], 2, &[1, heads, ctx, hd]);
        let vnew = b.concat(&[&vv, &v], 2, &[1, heads, ctx, hd]);
        kv_out.push(knew.clone());
        kv_out.push(vnew.clone());

        let att = b.matmul(&q, &knew, true, &[1, heads, steps, ctx]);
        let sc = b.scalar_f32((hd as f32).powf(-0.5));
        let att = b.mul(&att, &sc, &[1, heads, steps, ctx]);
        let att = b.add(&att, &mask, &[1, heads, steps, ctx]);
        let att = b.softmax(&att, -1);
        let o = b.matmul(&att, &vnew, false, &[1, heads, steps, hd]);
        let o = b.transpose(&o, &[0, 2, 1, 3], &[1, steps, heads, hd]);
        let o = b.reshape(&o, &[1, steps, d]);
        let ow = w(&mut b, wt, &format!("{p}.self_attn.out_proj.weight"))?;
        let o = b.linear(&o, &ow, None, &[1, steps, d]);
        let ls1 = w(&mut b, wt, &format!("{p}.layer_scale_1.scale"))?;
        let o = b.mul(&o, &ls1, &[1, steps, d]);
        h = b.add(&h, &o, &[1, steps, d]);

        let n2w = w(&mut b, wt, &format!("{p}.norm2.weight"))?;
        let n2b = w(&mut b, wt, &format!("{p}.norm2.bias"))?;
        let n = b.layer_norm(&h, &n2w, &n2b, 1e-5);
        let l1 = w(&mut b, wt, &format!("{p}.linear1.weight"))?;
        let l2 = w(&mut b, wt, &format!("{p}.linear2.weight"))?;
        let f = b.linear(&n, &l1, None, &[1, steps, dm.ff]);
        let f = b.gelu(&f, gelu);
        let f = b.linear(&f, &l2, None, &[1, steps, d]);
        let ls2 = w(&mut b, wt, &format!("{p}.layer_scale_2.scale"))?;
        let f = b.mul(&f, &ls2, &[1, steps, d]);
        h = b.add(&h, &f, &[1, steps, d]);
    }
    if d != dm.out_dim {
        let opw = w(&mut b, wt, "mimi.decoder_transformer.output_proj.0.weight")?;
        h = b.linear(&h, &opw, None, &[1, steps, dm.out_dim]);
    }
    let mut x = b.transpose(&h, &[0, 2, 1], &[1, dm.out_dim, steps]);

    // SEANet: a conv, then per ratio a transposed conv and a residual block, then a last conv
    let dec = "mimi.decoder.model";
    let mut ch = dm.n_filters << dm.ratios.len();
    let c0w = w(&mut b, wt, &format!("{dec}.0.conv.weight"))?;
    let c0b = w(&mut b, wt, &format!("{dec}.0.conv.bias"))?;
    let k = dm.kernel_size;
    let (y, c0_o) = sconv(&mut b, &x, &conv_st[1], &c0w, &c0b, dm.out_dim, ch, steps, k, k - 1);
    x = y;
    let mut conv_out = vec![up_o, c0_o];
    let mut t = steps;
    for (i, &r) in dm.ratios.iter().enumerate() {
        let (ti, ri) = (3 * i + 2, 3 * i + 3);
        let tw = w(&mut b, wt, &format!("{dec}.{ti}.convtr.weight"))?;
        let tb = w(&mut b, wt, &format!("{dec}.{ti}.convtr.bias"))?;
        let e = b.elu(&x);
        ch /= 2;
        let (y, t_o) = sconvtr(&mut b, &e, &conv_st[2 + 2 * i], &tw, Some(&tb), ch, t, r, 2 * r, 1);
        t *= r;
        let (mid, rk) = (ch / dm.compress, dm.residual_kernel_size);
        let (y, r_o) = resblock(&mut b, wt, ri, &y, &conv_st[3 + 2 * i], ch, mid, t, rk)?;
        x = y;
        conv_out.extend([t_o, r_o]);
    }
    let last = 3 * dm.ratios.len() + 2;
    let x = b.elu(&x);
    let cw = w(&mut b, wt, &format!("{dec}.{last}.conv.weight"))?;
    let cb = w(&mut b, wt, &format!("{dec}.{last}.conv.bias"))?;
    let lk = dm.last_kernel_size;
    let (pcm, c_o) = sconv(&mut b, &x, &conv_st[conv_st.len() - 1], &cw, &cb, ch, 1, t, lk, lk - 1);
    conv_out.push(c_o);

    // Name the outputs: state has to be carried back in by name, and CoreML's feature-name
    // set is unordered.
    let pcm = b.alias(&pcm, "pcm");
    let named: Vec<Var> = conv_out
        .iter()
        .chain(kv_out.iter())
        .zip(spec.iter())
        .map(|(v, (n, _))| b.alias(v, &format!("{n}_out")))
        .collect();
    let mut outs: Vec<&Var> = vec![&pcm];
    outs.extend(named.iter());
    Ok(b.finish_with_weights(&outs))
}

fn rope(
    b: &mut Builder,
    x: &Var,
    cos: &Var,
    sin: &Var,
    heads: usize,
    steps: usize,
    hd: usize,
) -> Var {
    let pr = hd / 2;
    let (h, t) = (heads as i32, steps as i32);
    let xr = b.reshape(x, &[1, heads, steps, pr, 2]);
    let lo = b.slice(&xr, &[0, 0, 0, 0, 0], &[1, h, t, pr as i32, 1], &[1, heads, steps, pr, 1]);
    let hi = b.slice(&xr, &[0, 0, 0, 0, 1], &[1, h, t, pr as i32, 2], &[1, heads, steps, pr, 1]);
    let nh = b.neg(&hi);
    let r = b.concat(&[&nh, &lo], 4, &[1, heads, steps, pr, 2]);
    let r = b.reshape(&r, &[1, heads, steps, hd]);
    let a = b.mul(x, cos, &[1, heads, steps, hd]);
    let c = b.mul(&r, sin, &[1, heads, steps, hd]);
    b.add(&a, &c, &[1, heads, steps, hd])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_dims_keep_the_buffers_older_bundles_take() {
        let spec = Dims::default().state_spec(266);
        let kv = vec![1, 8, 266, 64];
        let want: Vec<(&str, Vec<usize>)> = vec![
            ("up_partial", vec![1, 512, 16]),
            ("c0_prev", vec![1, 512, 6]),
            ("t2_partial", vec![1, 256, 6]),
            ("r3_prev", vec![1, 256, 2]),
            ("t5_partial", vec![1, 128, 5]),
            ("r6_prev", vec![1, 128, 2]),
            ("t8_partial", vec![1, 64, 4]),
            ("r9_prev", vec![1, 64, 2]),
            ("c11_prev", vec![1, 64, 2]),
            ("k0", kv.clone()),
            ("v0", kv.clone()),
            ("k1", kv.clone()),
            ("v1", kv),
        ];
        let got: Vec<(&str, Vec<usize>)> =
            spec.iter().map(|(n, s)| (n.as_str(), s.clone())).collect();
        assert_eq!(got, want);
        assert_eq!(Dims::default().frame(), 1920);
    }

    #[test]
    fn dims_round_trip_through_bundle_json() {
        let dm = Dims {
            d_model: 128,
            heads: 4,
            layers: 12,
            ff: 384,
            out_dim: 128,
            n_filters: 32,
            kernel_size: 3,
            ..Dims::default()
        };
        assert_eq!(Dims::from_json(&dm.to_json()).unwrap(), dm);
        assert_eq!(Dims::from_json(&serde_json::Value::Null).unwrap(), Dims::default());
    }
}
