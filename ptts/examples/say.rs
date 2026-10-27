//! The shortest thing that makes a sound: the default model, or the one `--model` names.
//!
//! ```text
//! cargo run --release --example say --features hf,hub -- "hello world"
//! cargo run --release --example say --features hf -- --model /path/to/model "hello world"
//! ```

use anyhow::Context as _;
use ptts::checkpoint::{Checkpoint, DEFAULT_MODEL, ResolveOptions};
use ptts::preprocess::Lang;
use ptts::synth::{DeviceKind, Quant};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut model = DEFAULT_MODEL.to_string();
    let mut text = args.next();
    if text.as_deref() == Some("--model") {
        model =
            args.next().context("usage: say [--model <directory or hf://owner/name>] [text]")?;
        text = args.next();
    }
    let text = text.unwrap_or_else(|| "Hello from Phonon.".to_string());
    let checkpoint =
        Checkpoint::locate(&model, ResolveOptions { quant: Quant::Q80, weights: None })?;
    let mut tts = checkpoint.builder(Lang::En).device(DeviceKind::Cpu).build()?;
    checkpoint.register_voices(&mut tts);
    let pcm = tts.say(&text)?;
    ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate())?;
    println!("wrote out.wav ({:.2}s)", pcm.len() as f32 / tts.sample_rate() as f32);
    Ok(())
}
