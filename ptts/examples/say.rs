//! The shortest thing that makes a sound.
//!
//! ```text
//! cargo run --release --example say --features sp,hub -- "hello world"
//! ```

#[path = "model_helpers.rs"]
mod model_helpers;

fn main() -> anyhow::Result<()> {
    let text = std::env::args().nth(1).unwrap_or_else(|| "Hello from Pocket TTS.".to_string());

    let tts = ptts::synth::Synth::from_pretrained(model_helpers::REPO_ID)?;
    let pcm = tts.say(&text)?;
    ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate() as u32)?;

    println!("wrote out.wav ({:.2}s)", pcm.len() as f32 / tts.sample_rate() as f32);
    Ok(())
}
