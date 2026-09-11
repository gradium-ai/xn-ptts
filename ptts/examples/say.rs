//! The shortest thing that makes a sound.
//!
//! ```text
//! cargo run --release --example say --features sp -- "hello world"
//! ```

#[path = "model_helpers.rs"]
mod model_helpers;

fn main() -> anyhow::Result<()> {
    let text = std::env::args().nth(1).unwrap_or_else(|| "Hello from Pocket TTS.".to_string());

    // `ptts` reads the files it is handed; finding them is the frontend's job,
    // and for the examples `model_helpers` is where that knowledge lives.
    let checkpoint = model_helpers::Checkpoint::from_hub(model_helpers::REPO_ID)?;
    let mut tts = checkpoint.builder().build()?;
    checkpoint.register_voices(&mut tts);

    let pcm = tts.say_with(&text, &ptts::synth::SpeechOptions::default().voice("alba"))?;
    ptts::wav::write_wav_file("out.wav", &pcm, tts.sample_rate() as u32)?;

    println!("wrote out.wav ({:.2}s)", pcm.len() as f32 / tts.sample_rate() as f32);
    Ok(())
}
