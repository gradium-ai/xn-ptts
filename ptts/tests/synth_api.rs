//! Exercises everything about `Synth` that does not need model weights.
//!
//! The generation path itself needs a checkpoint, so it is covered by the
//! examples rather than here. What is testable without one is the surface most
//! likely to break for a first-time user: argument parsing, feature gating, and
//! the errors on the paths they will hit by accident.
//!
//! Those errors are asserted on as values -- the variant -- as well as by message. A message is
//! allowed to be reworded; which class of failure it is, is the contract.

use ptts::Error;
use ptts::preprocess::{Lang, Normalize};
use ptts::synth::{DeviceKind, Quant, SpeechOptions, Synth, SynthBuilder};

/// A builder over `weights`, with the shipped config: every test here fails
/// before the weights are read, so the config's contents do not matter, and
/// neither does which language it would have normalized as.
fn builder(weights: impl Into<std::path::PathBuf>) -> SynthBuilder {
    Synth::builder(
        serde_json::from_str(include_str!("fixtures/config.json")).unwrap(),
        weights,
        Normalize::for_lang(Lang::En),
    )
}

#[test]
fn quant_parses_every_spelling_the_clis_accept() {
    let cases = [
        ("f32", Quant::F32),
        ("none", Quant::F32),
        ("q8", Quant::Q80),
        ("q8_0", Quant::Q80),
        ("q8_1", Quant::Q81),
        ("q8k", Quant::Q8k),
        ("q6k", Quant::Q6k),
        ("q5", Quant::Q50),
        ("q5_0", Quant::Q50),
        ("q5_1", Quant::Q51),
        ("q5k", Quant::Q5k),
        ("q4", Quant::Q40),
        ("q4_0", Quant::Q40),
        ("q4_1", Quant::Q41),
        ("q4k", Quant::Q4k),
    ];
    for (name, expected) in cases {
        assert_eq!(name.parse::<Quant>().unwrap(), expected, "parsing {name}");
    }
}

#[test]
fn quant_round_trips_through_its_canonical_name() {
    for quant in [
        Quant::F32,
        Quant::Q80,
        Quant::Q81,
        Quant::Q8k,
        Quant::Q6k,
        Quant::Q50,
        Quant::Q51,
        Quant::Q5k,
        Quant::Q40,
        Quant::Q41,
        Quant::Q4k,
    ] {
        assert_eq!(quant.as_str().parse::<Quant>().unwrap(), quant);
    }
    assert_eq!("Q8_0".parse::<Quant>().unwrap(), Quant::Q80, "any case");
}

#[test]
fn unknown_quant_lists_the_valid_ones() {
    let err = "q3k".parse::<Quant>().unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    let err = err.to_string();
    assert!(err.contains("q3k"), "{err}");
    assert!(err.contains("q4k"), "should list the supported formats: {err}");
}

#[test]
fn device_parses_and_rejects_unknown_names() {
    assert_eq!("auto".parse::<DeviceKind>().unwrap(), DeviceKind::Auto);
    assert_eq!("cpu".parse::<DeviceKind>().unwrap(), DeviceKind::Cpu);
    assert_eq!(" CPU ".parse::<DeviceKind>().unwrap(), DeviceKind::Cpu, "trimmed, any case");
    assert_eq!("cuda".parse::<DeviceKind>().unwrap(), DeviceKind::Cuda);
    assert_eq!("vulkan".parse::<DeviceKind>().unwrap(), DeviceKind::Vulkan);
    assert_eq!("metal".parse::<DeviceKind>().unwrap(), DeviceKind::Metal);
    let err = "tpu".parse::<DeviceKind>().unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    assert!(err.to_string().contains("tpu"), "{err}");
}

#[test]
fn auto_resolves_to_something_concrete() {
    let resolved = DeviceKind::Auto.resolve();
    assert_ne!(resolved, DeviceKind::Auto);
    // With no GPU feature enabled, `auto` must land on the CPU.
    if !cfg!(any(feature = "cuda", feature = "vulkan", feature = "metal")) {
        assert_eq!(resolved, DeviceKind::Cpu);
    }
}

#[test]
fn explicit_device_survives_resolution() {
    for device in [DeviceKind::Cpu, DeviceKind::Cuda, DeviceKind::Vulkan, DeviceKind::Metal] {
        assert_eq!(device.resolve(), device);
    }
}

#[test]
fn missing_weights_are_a_not_found_naming_the_path() {
    let err = builder("/definitely/not/a/model/weights.safetensors").build().unwrap_err();
    assert!(matches!(err, Error::NotFound(_)), "{err:?}");
    assert!(err.to_string().contains("/definitely/not/a/model/weights.safetensors"), "{err}");
}

#[test]
fn a_load_without_a_tokenizer_says_how_to_supply_one() {
    // Weights that do exist, so the load reaches the tokenizer step. The file
    // is not a checkpoint, but the tokenizer is resolved before it is read.
    let weights = std::env::temp_dir().join("ptts-synth-api-no-tokenizer.safetensors");
    std::fs::write(&weights, b"").unwrap();
    let err = builder(&weights).build().unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("SynthBuilder::tokenizer"), "{err}");
    std::fs::remove_file(&weights).ok();
}

#[test]
fn quantization_on_a_gpu_is_rejected_before_the_weights_are_touched() {
    let err = builder("/definitely/not/a/model/weights.safetensors")
        .device(DeviceKind::Cuda)
        .quant(Quant::Q40)
        .build()
        .unwrap_err();
    // Not `NotFound`: the check has to happen before the weights are opened, and the variant is
    // what proves the order rather than which words came out.
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    let err = err.to_string();
    assert!(err.contains("runs on the CPU"), "{err}");
    assert!(err.contains("q4_0"), "the error should name the format: {err}");
}

#[test]
fn unavailable_f32_backends_fail_before_reading_model_files() {
    for (device, name, compiled) in [
        (DeviceKind::Cuda, "cuda", cfg!(feature = "cuda")),
        (DeviceKind::Vulkan, "vulkan", cfg!(feature = "vulkan")),
        (DeviceKind::Metal, "metal", cfg!(feature = "metal")),
    ] {
        if compiled {
            continue;
        }
        let err = builder("/definitely/not/a/model/weights.safetensors")
            .device(device)
            .build()
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
        assert!(err.to_string().contains(name), "{err}");
        assert!(err.to_string().contains("not available in this build"), "{err}");
        let message = err.to_string();
        let available = message.split("available devices: ").nth(1).unwrap();
        assert!(available.split(", ").any(|d| d == "cpu"), "{message}");
        assert!(!available.split(", ").any(|d| d == name), "{message}");
    }
}

#[test]
fn preflight_accepts_compiled_backends_without_initializing_them() {
    Quant::F32.check_device(DeviceKind::Auto).unwrap();
    Quant::Q80.check_device(DeviceKind::Cpu).unwrap();
    for (device, compiled) in [
        (DeviceKind::Cpu, true),
        (DeviceKind::Cuda, cfg!(feature = "cuda")),
        (DeviceKind::Vulkan, cfg!(feature = "vulkan")),
        (DeviceKind::Metal, cfg!(feature = "metal")),
    ] {
        if compiled {
            Quant::F32.check_device(device).unwrap();
        }
    }
}

#[test]
fn speech_options_build_up_fluently() {
    let opts = SpeechOptions::default()
        .voice("alba")
        .temperature(0.5)
        .seed(7)
        .cfg_coef(1.5)
        .max_tokens_per_chunk(32);
    assert_eq!(opts.voice.as_deref(), Some("alba"));
    assert_eq!(opts.temperature, Some(0.5));
    assert_eq!(opts.seed, Some(7));
    assert_eq!(opts.cfg_coef, Some(1.5));
    assert_eq!(opts.max_tokens_per_chunk, Some(32));
}

#[test]
fn speech_options_default_to_the_builders_settings() {
    let opts = SpeechOptions::default();
    assert!(opts.voice.is_none());
    assert!(opts.temperature.is_none());
    assert!(opts.seed.is_none());
    assert!(opts.cfg_coef.is_none());
}

/// A session is the fix for re-priming the voice on every call, so the type has
/// to be nameable and its budget visible without a model in hand.
#[test]
fn session_is_part_of_the_public_api() {
    fn _accepts(_: &ptts::synth::Session) {}
    fn _budget(s: &ptts::synth::Session) -> usize {
        s.seq_budget()
    }
}

/// `Session` needs `Sync` as well as `Send`: `ptts-ws-server` holds a
/// `&Session` across an await, and tokio requires that future to be `Send`.
/// That does **not** make concurrent generation from one session safe — see the
/// note on `Session`, which the type cannot enforce. `SpeechStream` is `Send`
/// only: it owns an mpsc receiver, which is why `ptts-pyo3` wraps it in a mutex.
#[test]
fn the_types_the_frontends_move_between_threads_still_can() {
    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}
    // `ptts-ws-server` is built on `anyhow`, which only accepts a `Send + Sync` error.
    send_sync::<ptts::Error>();
    send_sync::<ptts::synth::Synth>();
    send_sync::<ptts::synth::Session>();
    send::<ptts::synth::SpeechStream>();
}

#[test]
fn a_frontend_on_xn_result_still_compiles() {
    // The compatibility bridge: `?` on this crate inside a function returning `xn::Result` is
    // what every existing frontend does, and it has to keep working.
    fn _frontend() -> xn::Result<()> {
        let _ = "q8_0".parse::<Quant>()?;
        let _ = "cpu".parse::<DeviceKind>()?;
        Ok(())
    }
}
