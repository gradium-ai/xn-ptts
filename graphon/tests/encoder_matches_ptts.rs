//! Gemimi's encoder transformer against ptts's Mimi transformer, which it reimplements with a
//! ring-buffer cache. Needs an export: `GRAPHON_MODEL=<export dir> cargo test -r -p graphon`.

use graphon::config::Config;
use graphon::gemimi::Gemimi;
use ptts::transformer::ProjectedTransformer;
use xn::Tensor;
use xn::nn::VB;

type F32 = xn::Unquantized<f32, xn::CpuDevice>;

#[test]
fn encoder_transformer_matches_ptts() -> xn::Result<()> {
    let Ok(dir) = std::env::var("GRAPHON_MODEL") else {
        eprintln!("GRAPHON_MODEL is not set, skipping");
        return Ok(());
    };
    let dir = std::path::Path::new(&dir);
    let cfg = Config::load(&dir.join("config.json"))?.gemimi;
    let vb = VB::load(&[dir.join("gemimi.safetensors")], xn::CPU)?.root();
    let gemimi = Gemimi::<F32>::load(&vb, &cfg)?;
    let d = cfg.transformer_d_model;
    let reference = ProjectedTransformer::<F32>::load(
        &vb.pp("encoder_transformer"),
        cfg.dimension,
        &[d],
        d,
        cfg.transformer_num_heads,
        cfg.transformer_num_layers,
        Some(0.01),
        cfg.transformer_context,
        cfg.transformer_max_period,
        cfg.transformer_dim_feedforward,
    )?;
    let mut state = gemimi.init_state(1)?;
    let mut ref_state = reference.init_state(1, 0)?;
    let frame = gemimi.frame_size();
    // Two transformer frames per step: 80 steps run well past the attention window.
    let mut worst = 0f32;
    for step in 0..80 {
        let pcm = (0..frame)
            .map(|i| {
                let n = step * frame + i;
                0.1 * (n as f32 * 0.031).sin() + 0.05 * ((n * 7919 % 1000) as f32 / 500.0 - 1.0)
            })
            .collect();
        let pcm = Tensor::from_vec(pcm, (1, 1, frame), &xn::CPU)?;
        let emb = gemimi.seanet(&pcm, &mut state)?;
        let ours = gemimi.transformer(&emb, &mut state)?.to_vec()?;
        let theirs = reference.forward(&emb, &mut ref_state)?.swap_remove(0);
        let theirs = theirs.transpose(1, 2)?.contiguous()?.to_vec()?;
        let diff = ours.iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let scale = theirs.iter().map(|b| b.abs()).fold(0f32, f32::max);
        worst = worst.max(diff / scale);
    }
    eprintln!("worst relative difference {worst:.2e}");
    assert!(worst < 1e-4, "worst relative difference {worst}");
    Ok(())
}
