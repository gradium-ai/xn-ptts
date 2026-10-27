//! Build the Mimi decoder graph.
//!
//! One Mimi frame: `latent -> output_proj -> stride-16 depthwise upsample -> 2-layer
//! transformer at 200 Hz -> SEANet (x6, x5, x4 transposed convs) -> 1920 PCM samples`.
//!
//! Streaming state travels as ordinary inputs and outputs rather than CoreML state, because
//! CoreML states are fp16-only and Mimi has to run in f32. It is ~1.05MB each way, almost all
//! of it the transformer KV cache.

use crate::mil::{Builder, DType, Var};
use crate::weights::Weights;

pub const DIM: usize = 512;
pub const HEADS: usize = 8;
pub const HD: usize = DIM / HEADS;
pub const LAYERS: usize = 2;
pub const FF: usize = 2048;
pub const LDIM: usize = 32;
pub const UP_STRIDE: usize = 16;
pub const STEPS: usize = UP_STRIDE;
pub const RATIOS: [usize; 3] = [6, 5, 4];
pub const FRAME: usize = 1920;

/// Slots of the transformer KV cache for an attention window of `window` positions.
///
/// The reference keeps `window - 1` positions between calls and appends the call's `STEPS` new
/// rows before attending, so every query sees `window` positions, itself included. This cache
/// keeps one more, which the mask never shows, so bundles exported with it still load. A cache
/// of only `window` slots, new rows included, would drop up to `STEPS` of the oldest ones.
pub fn cache_len(window: usize) -> usize {
    window + STEPS
}

/// Package name of the Mimi graph for a window, so a cached graph of another cache length is
/// never picked up in its place.
pub fn package_name(window: usize) -> String {
    format!("mimi-c{}", cache_len(window))
}

/// Every streaming buffer, in the order the graph takes and returns them. `ctx` is the cache
/// length, `cache_len(window)`.
pub fn state_spec(ctx: usize) -> Vec<(String, Vec<usize>)> {
    let mut v: Vec<(String, Vec<usize>)> = vec![
        ("up_partial".into(), vec![1, DIM, UP_STRIDE]),
        ("c0_prev".into(), vec![1, DIM, 6]),
        ("t2_partial".into(), vec![1, 256, 6]),
        ("r3_prev".into(), vec![1, 256, 2]),
        ("t5_partial".into(), vec![1, 128, 5]),
        ("r6_prev".into(), vec![1, 128, 2]),
        ("t8_partial".into(), vec![1, 64, 4]),
        ("r9_prev".into(), vec![1, 64, 2]),
        ("c11_prev".into(), vec![1, 64, 2]),
    ];
    for i in 0..LAYERS {
        v.push((format!("k{i}"), vec![1, HEADS, ctx, HD]));
        v.push((format!("v{i}"), vec![1, HEADS, ctx, HD]));
    }
    v
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
) -> Result<(Var, Var), String> {
    let p = format!("mimi.decoder.model.{idx}");
    let w1 = w(b, wt, &format!("{p}.block.1.conv.weight"))?;
    let b1 = w(b, wt, &format!("{p}.block.1.conv.bias"))?;
    let w3 = w(b, wt, &format!("{p}.block.3.conv.weight"))?;
    let b3 = w(b, wt, &format!("{p}.block.3.conv.bias"))?;
    let h = b.elu(x);
    let (h, tail) = sconv(b, &h, prev, &w1, &b1, cin, mid, t, 3, 2);
    let h = b.elu(&h);
    let h = b.conv1d(&h, &w3, Some(&b3), 1, &[1, cin, t]);
    Ok((b.add(x, &h, &[1, cin, t]), tail))
}

/// The Mimi graph for one frame, with a `ctx`-slot transformer cache (`cache_len(window)`).
///
/// f32 throughout, so it runs on the CPU: the vocoder is kept at its original precision, and
/// the Neural Engine is fp16-only.
pub fn build(wt: &Weights, ctx: usize) -> Result<super::flow_lm::Built, String> {
    let fp = DType::Fp32;
    let mut b = Builder::new().with_float(fp);
    let latent = b.input("latent", fp, &[1, 1, LDIM]);
    let cos = b.input("cos", fp, &[1, 1, STEPS, HD]);
    let sin = b.input("sin", fp, &[1, 1, STEPS, HD]);
    let mask = b.input("mask", fp, &[1, 1, STEPS, ctx]);
    let spec = state_spec(ctx);
    let st: Vec<Var> = spec.iter().map(|(n, sh)| b.input(n, fp, sh)).collect();

    // denormalise, then 32 -> 512
    let mean = w(&mut b, wt, "flow_lm.emb_mean")?;
    let sd = w(&mut b, wt, "flow_lm.emb_std")?;
    let x = b.mul(&latent, &sd, &[1, 1, LDIM]);
    let x = b.add(&x, &mean, &[1, 1, LDIM]);
    let x = b.transpose(&x, &[0, 2, 1], &[1, LDIM, 1]);
    let opw = w(&mut b, wt, "mimi.quantizer.output_proj.weight")?;
    let x = b.conv1d(&x, &opw, None, 1, &[1, DIM, 1]);

    // stride-16 depthwise upsample: 12.5 Hz -> 200 Hz
    let upw = w(&mut b, wt, "mimi.upsample.convtr.convtr.weight")?;
    let (x, up_o) = sconvtr(&mut b, &x, &st[0], &upw, None, DIM, 1, UP_STRIDE, 2 * UP_STRIDE, DIM);

    // transformer over the 16 new positions
    let mut h = b.transpose(&x, &[0, 2, 1], &[1, STEPS, DIM]);
    let mut kv_out = Vec::new();
    for i in 0..LAYERS {
        let p = format!("mimi.decoder_transformer.transformer.layers.{i}");
        let n1w = w(&mut b, wt, &format!("{p}.norm1.weight"))?;
        let n1b = w(&mut b, wt, &format!("{p}.norm1.bias"))?;
        let n = b.layer_norm(&h, &n1w, &n1b, 1e-5);
        let inw = w(&mut b, wt, &format!("{p}.self_attn.in_proj.weight"))?;
        let qkv = b.linear(&n, &inw, None, &[1, STEPS, 3 * DIM]);
        let qkv = b.reshape(&qkv, &[1, STEPS, 3, HEADS, HD]);
        let qkv = b.transpose(&qkv, &[2, 0, 3, 1, 4], &[3, 1, HEADS, STEPS, HD]);
        let part = |b: &mut Builder, k: i32| {
            let s = b.slice(
                &qkv,
                &[k, 0, 0, 0, 0],
                &[k + 1, 1, HEADS as i32, STEPS as i32, HD as i32],
                &[1, 1, HEADS, STEPS, HD],
            );
            b.reshape(&s, &[1, HEADS, STEPS, HD])
        };
        let q = part(&mut b, 0);
        let k = part(&mut b, 1);
        let v = part(&mut b, 2);
        let q = rope(&mut b, &q, &cos, &sin);
        let k = rope(&mut b, &k, &cos, &sin);

        let kin = &st[9 + 2 * i];
        let vin = &st[10 + 2 * i];
        let kk = b.slice(
            kin,
            &[0, 0, STEPS as i32, 0],
            &[1, HEADS as i32, ctx as i32, HD as i32],
            &[1, HEADS, ctx - STEPS, HD],
        );
        let vv = b.slice(
            vin,
            &[0, 0, STEPS as i32, 0],
            &[1, HEADS as i32, ctx as i32, HD as i32],
            &[1, HEADS, ctx - STEPS, HD],
        );
        let knew = b.concat(&[&kk, &k], 2, &[1, HEADS, ctx, HD]);
        let vnew = b.concat(&[&vv, &v], 2, &[1, HEADS, ctx, HD]);
        kv_out.push(knew.clone());
        kv_out.push(vnew.clone());

        let att = b.matmul(&q, &knew, true, &[1, HEADS, STEPS, ctx]);
        let sc = b.scalar_f32((HD as f32).powf(-0.5));
        let att = b.mul(&att, &sc, &[1, HEADS, STEPS, ctx]);
        let att = b.add(&att, &mask, &[1, HEADS, STEPS, ctx]);
        let att = b.softmax(&att, -1);
        let o = b.matmul(&att, &vnew, false, &[1, HEADS, STEPS, HD]);
        let o = b.transpose(&o, &[0, 2, 1, 3], &[1, STEPS, HEADS, HD]);
        let o = b.reshape(&o, &[1, STEPS, DIM]);
        let ow = w(&mut b, wt, &format!("{p}.self_attn.out_proj.weight"))?;
        let o = b.linear(&o, &ow, None, &[1, STEPS, DIM]);
        let ls1 = w(&mut b, wt, &format!("{p}.layer_scale_1.scale"))?;
        let o = b.mul(&o, &ls1, &[1, STEPS, DIM]);
        h = b.add(&h, &o, &[1, STEPS, DIM]);

        let n2w = w(&mut b, wt, &format!("{p}.norm2.weight"))?;
        let n2b = w(&mut b, wt, &format!("{p}.norm2.bias"))?;
        let n = b.layer_norm(&h, &n2w, &n2b, 1e-5);
        let l1 = w(&mut b, wt, &format!("{p}.linear1.weight"))?;
        let l2 = w(&mut b, wt, &format!("{p}.linear2.weight"))?;
        let f = b.linear(&n, &l1, None, &[1, STEPS, FF]);
        let f = b.gelu_exact(&f);
        let f = b.linear(&f, &l2, None, &[1, STEPS, DIM]);
        let ls2 = w(&mut b, wt, &format!("{p}.layer_scale_2.scale"))?;
        let f = b.mul(&f, &ls2, &[1, STEPS, DIM]);
        h = b.add(&h, &f, &[1, STEPS, DIM]);
    }
    let x = b.transpose(&h, &[0, 2, 1], &[1, DIM, STEPS]);

    // SEANet
    let d = "mimi.decoder.model";
    let c0w = w(&mut b, wt, &format!("{d}.0.conv.weight"))?;
    let c0b = w(&mut b, wt, &format!("{d}.0.conv.bias"))?;
    let (x, c0_o) = sconv(&mut b, &x, &st[1], &c0w, &c0b, DIM, DIM, STEPS, 7, 6);
    let x = b.elu(&x);

    let t2w = w(&mut b, wt, &format!("{d}.2.convtr.weight"))?;
    let t2b = w(&mut b, wt, &format!("{d}.2.convtr.bias"))?;
    let (x, t2_o) = sconvtr(&mut b, &x, &st[2], &t2w, Some(&t2b), 256, STEPS, RATIOS[0], 12, 1);
    let t_a = STEPS * RATIOS[0];
    let (x, r3_o) = resblock(&mut b, wt, 3, &x, &st[3], 256, 128, t_a)?;
    let x = b.elu(&x);

    let t5w = w(&mut b, wt, &format!("{d}.5.convtr.weight"))?;
    let t5b = w(&mut b, wt, &format!("{d}.5.convtr.bias"))?;
    let (x, t5_o) = sconvtr(&mut b, &x, &st[4], &t5w, Some(&t5b), 128, t_a, RATIOS[1], 10, 1);
    let t_b = t_a * RATIOS[1];
    let (x, r6_o) = resblock(&mut b, wt, 6, &x, &st[5], 128, 64, t_b)?;
    let x = b.elu(&x);

    let t8w = w(&mut b, wt, &format!("{d}.8.convtr.weight"))?;
    let t8b = w(&mut b, wt, &format!("{d}.8.convtr.bias"))?;
    let (x, t8_o) = sconvtr(&mut b, &x, &st[6], &t8w, Some(&t8b), 64, t_b, RATIOS[2], 8, 1);
    let t_c = t_b * RATIOS[2];
    let (x, r9_o) = resblock(&mut b, wt, 9, &x, &st[7], 64, 32, t_c)?;
    let x = b.elu(&x);

    let c11w = w(&mut b, wt, &format!("{d}.11.conv.weight"))?;
    let c11b = w(&mut b, wt, &format!("{d}.11.conv.bias"))?;
    let (pcm, c11_o) = sconv(&mut b, &x, &st[8], &c11w, &c11b, 64, 1, t_c, 3, 2);

    // Name the outputs: state has to be carried back in by name, and CoreML's feature-name
    // set is unordered.
    let pcm = b.alias(&pcm, "pcm");
    let named: Vec<Var> = [&up_o, &c0_o, &t2_o, &r3_o, &t5_o, &r6_o, &t8_o, &r9_o, &c11_o]
        .iter()
        .zip(spec.iter())
        .map(|(v, (n, _))| b.alias(v, &format!("{n}_out")))
        .collect();
    let kv_named: Vec<Var> = kv_out
        .iter()
        .zip(spec.iter().skip(9))
        .map(|(v, (n, _))| b.alias(v, &format!("{n}_out")))
        .collect();
    let mut outs: Vec<&Var> = vec![&pcm];
    outs.extend(named.iter());
    outs.extend(kv_named.iter());
    Ok(b.finish_with_weights(&outs))
}

fn rope(b: &mut Builder, x: &Var, cos: &Var, sin: &Var) -> Var {
    let pr = HD / 2;
    let xr = b.reshape(x, &[1, HEADS, STEPS, pr, 2]);
    let lo = b.slice(
        &xr,
        &[0, 0, 0, 0, 0],
        &[1, HEADS as i32, STEPS as i32, pr as i32, 1],
        &[1, HEADS, STEPS, pr, 1],
    );
    let hi = b.slice(
        &xr,
        &[0, 0, 0, 0, 1],
        &[1, HEADS as i32, STEPS as i32, pr as i32, 2],
        &[1, HEADS, STEPS, pr, 1],
    );
    let nh = b.neg(&hi);
    let r = b.concat(&[&nh, &lo], 4, &[1, HEADS, STEPS, pr, 2]);
    let r = b.reshape(&r, &[1, HEADS, STEPS, HD]);
    let a = b.mul(x, cos, &[1, HEADS, STEPS, HD]);
    let c = b.mul(&r, sin, &[1, HEADS, STEPS, HD]);
    b.add(&a, &c, &[1, HEADS, STEPS, HD])
}
