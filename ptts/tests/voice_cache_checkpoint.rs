//! Opt-in real-checkpoint regression checks; no network access or downloads.
#![cfg(feature = "hf")]

use ptts::checkpoint::{Checkpoint, ResolveOptions};
use ptts::preprocess::Lang;
use ptts::synth::{DeviceKind, Quant, SpeechOptions, Synth, VoiceCacheConfig};

fn synth(config: VoiceCacheConfig) -> Synth {
    let path =
        std::path::PathBuf::from(std::env::var("PTTS_TEST_MODEL").expect("set PTTS_TEST_MODEL"));
    let checkpoint =
        Checkpoint::resolve(&path, ResolveOptions { quant: Quant::Q80, weights: None }).unwrap();
    // Two registered names deliberately use the same embedding to isolate cache behavior
    // from voice/model differences. The public benchmark uses distinct catalog voices.
    checkpoint
        .builder(Lang::En)
        .device(DeviceKind::Cpu)
        .quant(Quant::Q80)
        .voice_cache(config)
        .add_voice("a", path.join("default-voice.safetensors"))
        .add_voice("b", path.join("default-voice.safetensors"))
        .build()
        .unwrap()
}

fn assert_same_audio(a: &[f32], b: &[f32]) {
    assert!(!a.is_empty());
    assert_eq!(a.len(), b.len(), "cache policy must not change EOS or audio duration");
    assert!(a.iter().chain(b).all(|x| x.is_finite()));
    let max_error = a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_error < 1e-5, "max waveform difference {max_error}");
}

#[test]
#[ignore = "requires PTTS_TEST_MODEL; run with RAYON_NUM_THREADS=1 XN_KAI=0"]
fn eviction_and_bypass_preserve_audio_and_request_isolation() {
    let cached = synth(VoiceCacheConfig { max_entries: 1, ..Default::default() });
    let uncached = synth(VoiceCacheConfig { max_entries: 0, ..Default::default() });
    let text = "Hello from Phonon.";
    let opts = SpeechOptions::default().voice("a").seed(42);
    let reference = uncached.say_with(text, &opts).unwrap();
    assert_same_audio(&reference, &cached.say_with(text, &opts).unwrap());
    assert_same_audio(&reference, &cached.say_with(text, &opts).unwrap());
    assert_eq!(cached.voice_cache_stats().hits, 1);
    drop(cached.session(&SpeechOptions::default().voice("b"), 2048).unwrap());
    assert_eq!(cached.voice_cache_stats().evictions, 1);
    assert_same_audio(&reference, &cached.say_with(text, &opts).unwrap());
    assert_eq!(cached.voice_cache_stats().evictions, 2);
    assert!(cached.voice_cache_stats().kv_bytes > 0);
    assert_eq!(uncached.voice_cache_stats().entries, 0);
    assert_eq!(uncached.voice_cache_stats().bypasses, 1);

    let first = cached.session(&opts, 2048).unwrap();
    let second = cached.session(&opts, 2048).unwrap();
    let (a, b) = std::thread::scope(|scope| {
        let a = scope.spawn(|| first.say(text).unwrap());
        let b = scope.spawn(|| second.say(text).unwrap());
        (a.join().unwrap(), b.join().unwrap())
    });
    assert_same_audio(&reference, &a);
    assert_same_audio(&reference, &b);
}

#[test]
#[ignore = "requires PTTS_TEST_MODEL"]
fn replacing_a_voice_and_conditions_invalidates_retained_prefixes() {
    let mut cached = synth(Default::default());
    for name in ["a", "b"] {
        drop(cached.session(&SpeechOptions::default().voice(name), 2048).unwrap());
    }
    assert_eq!(cached.voice_cache_stats().entries, 2);
    let path = std::path::PathBuf::from(std::env::var("PTTS_TEST_MODEL").unwrap());
    cached.add_voice_file("a", &path.join("default-voice.safetensors")).unwrap();
    assert_eq!(cached.voice_cache_stats().entries, 1);
    assert_eq!(cached.voice_cache_stats().invalidations, 1);
    cached.set_conditions(Default::default()).unwrap();
    assert_eq!(cached.voice_cache_stats().entries, 0);
    assert_eq!(cached.voice_cache_stats().kv_bytes, 0);
    assert_eq!(cached.voice_cache_stats().invalidations, 2);
}
