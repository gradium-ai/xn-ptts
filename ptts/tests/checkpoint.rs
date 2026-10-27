//! File-resolution regressions. Fixtures contain no real model weights.
#![cfg(not(target_arch = "wasm32"))]

use ptts::Error;
use ptts::checkpoint::{Checkpoint, ResolveOptions, legacy_source};
use ptts::synth::Quant;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ptts-checkpoint-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let f = Self(std::fs::canonicalize(dir).unwrap());
        f.write("config.json", include_bytes!("fixtures/config.json"));
        f.write("tokenizer.json", b"{}");
        f
    }

    fn write(&self, name: &str, bytes: &[u8]) {
        let path = self.0.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    fn q8(&self, path: impl AsRef<Path>) -> ptts::Result<Checkpoint> {
        Checkpoint::resolve(path, ResolveOptions { quant: Quant::Q80, weights: None })
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_directory_and_its_config_resolve_identically() {
    let f = Fixture::new();
    f.write("model.safetensors", b"f32");
    f.write("model.q8.gguf", b"q8");
    f.write("voices/narrator.safetensors", b"voice");
    let dir = f.q8(&f.0).unwrap();
    let config = f.q8(f.0.join("config.json")).unwrap();
    assert_eq!(dir.weights, config.weights);
    assert_eq!(dir.weights.file_name().unwrap(), "model.q8.gguf");
    assert_eq!(dir.tokenizer, config.tokenizer);
    assert_eq!(dir.voices, config.voices);
    assert_eq!(Checkpoint::open(&f.0).unwrap().weights.file_name().unwrap(), "model.safetensors");
}

#[test]
fn checkpoints_require_their_own_config() {
    for name in ["model.safetensors", "model.q8.gguf", "custom.safetensors"] {
        let f = Fixture::new();
        std::fs::remove_file(f.0.join("config.json")).unwrap();
        f.write(name, b"weights");
        let err = f.q8(&f.0).unwrap_err();
        assert!(matches!(err, Error::NotFound(_)));
        assert!(err.to_string().contains("config.json"), "{err}");
    }
}

#[test]
fn nonstandard_weights_require_an_explicit_path() {
    let f = Fixture::new();
    f.write("weights/custom.safetensors", b"weights");
    assert!(matches!(f.q8(&f.0), Err(Error::NotFound(_))));
    let ck = Checkpoint::resolve(
        &f.0,
        ResolveOptions { quant: Quant::F32, weights: Some("weights/custom.safetensors") },
    )
    .unwrap();
    assert_eq!(ck.weights, f.0.join("weights/custom.safetensors"));
    for path in ["../weights", "/weights", "C:/weights", "a\\weights", "a//weights", "./weights"] {
        assert!(
            matches!(
                Checkpoint::resolve(
                    &f.0,
                    ResolveOptions { quant: Quant::F32, weights: Some(path) }
                ),
                Err(Error::InvalidData(_))
            ),
            "{path}"
        );
    }
}

#[test]
fn voice_precedence_is_identical_for_all_callers() {
    let f = Fixture::new();
    f.write("model.q8.gguf", b"q8");
    f.write("voices/default.safetensors", b"first");
    f.write("embeddings/default.safetensors", b"second");
    f.write("default-voice.safetensors", b"third");
    f.write("voices/z.safetensors", b"z");
    f.write("embeddings/a.safetensors", b"a");
    let ck = f.q8(&f.0).unwrap();
    assert_eq!(
        ck.voices,
        vec![
            ("a".into(), f.0.join("embeddings/a.safetensors")),
            ("default".into(), f.0.join("voices/default.safetensors")),
            ("z".into(), f.0.join("voices/z.safetensors"))
        ]
    );
}

#[test]
fn a_voice_catalog_adds_to_the_checkpoint_voices() {
    let f = Fixture::new();
    f.write("model.q8.gguf", b"q8");
    f.write("voices/a.safetensors", b"a");
    f.write("voices/b.safetensors", b"b");
    f.write(
        "catalog/voices.jsonl",
        br#"{"name": "b", "filename": "other-b.safetensors"}
{"name": "c", "filename": "c.safetensors"}
"#,
    );
    let mut ck = f.q8(&f.0).unwrap();
    ck.add_voice_catalog(f.0.join("catalog/voices.jsonl")).unwrap();
    assert_eq!(
        ck.voices,
        vec![
            ("a".into(), f.0.join("voices/a.safetensors")),
            ("b".into(), f.0.join("catalog/other-b.safetensors")),
            ("c".into(), f.0.join("catalog/c.safetensors"))
        ]
    );
}

#[test]
fn only_tokenizer_json_is_selected() {
    let f = Fixture::new();
    f.write("model.q8.gguf", b"q8");
    std::fs::remove_file(f.0.join("tokenizer.json")).unwrap();
    f.write("tokenizer.model", b"unused format");
    assert!(f.q8(&f.0).unwrap().tokenizer.is_none());
}

#[test]
fn malformed_configs_fail_without_a_fallback() {
    let f = Fixture::new();
    f.write("config.json", b"{}");
    f.write("model.q8.gguf", b"weights");
    let err = f.q8(&f.0).unwrap_err();
    assert!(matches!(err, Error::InvalidData(_)));
    assert!(err.to_string().contains("config.json"));
}

#[cfg(unix)]
#[test]
fn symlinked_configs_keep_the_snapshot_directory() {
    let f = Fixture::new();
    let blobs = Fixture::new();
    f.write("model.q8.gguf", b"q8");
    std::fs::remove_file(f.0.join("config.json")).unwrap();
    std::os::unix::fs::symlink(blobs.0.join("config.json"), f.0.join("config.json")).unwrap();
    for input in [&f.0, &f.0.join("config.json")] {
        let ck = f.q8(input).unwrap();
        assert_eq!(ck.weights, f.0.join("model.q8.gguf"));
        assert_eq!(ck.tokenizer, Some(f.0.join("tokenizer.json")));
    }
}

#[test]
fn only_hf_sources_are_hub_repos() {
    let opts = ResolveOptions::default();
    for path in ["./missing", "../missing", "/missing/checkpoint", "missing/config.json"] {
        assert!(matches!(Checkpoint::locate(path, opts), Err(Error::NotFound(_))), "{path}");
    }
    let Err(Error::NotFound(msg)) = Checkpoint::locate("owner/model", opts) else {
        panic!("a bare repo ID is a missing local path")
    };
    assert!(msg.contains("hf://owner/model"), "{msg}");
    assert!(matches!(Checkpoint::locate("hf://owner", opts), Err(Error::InvalidArgument(_))));
}

#[test]
fn the_default_model_is_a_hub_repo_pinned_to_a_commit() {
    let hub = ptts::hub::HubSource::parse(ptts::checkpoint::DEFAULT_MODEL).unwrap().unwrap();
    let revision = hub.revision.expect("the default names a revision");
    assert!(revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()), "{revision}");
}

#[test]
fn legacy_sources_keep_their_meaning() {
    for path in ["./missing", "../missing", "/missing/checkpoint", "missing/config.json"] {
        assert_eq!(legacy_source(path, None).unwrap(), path);
        assert!(matches!(legacy_source(path, Some("main")), Err(Error::InvalidArgument(_))));
    }
    assert_eq!(legacy_source("owner/model", None).unwrap(), "hf://owner/model");
    assert_eq!(legacy_source("owner/model", Some("abc")).unwrap(), "hf://owner/model@abc");
    assert_eq!(legacy_source("hf://owner/model@abc", None).unwrap(), "hf://owner/model@abc");
    let Err(Error::InvalidArgument(msg)) = legacy_source("hf://owner/model", Some("abc")) else {
        panic!("a separate revision for an hf:// source")
    };
    assert!(msg.contains("hf://owner/name@revision"), "{msg}");
}

#[cfg(windows)]
#[test]
fn windows_drive_rooted_paths_are_local_sources() {
    let path = r"\missing\checkpoint";
    assert!(Path::new(path).has_root());
    assert!(!Path::new(path).is_absolute());
    assert_eq!(legacy_source(path, None).unwrap(), path);
    assert!(matches!(Checkpoint::open(path), Err(Error::NotFound(_))));
}

#[test]
fn hub_voice_paths_use_the_checkpoint_names_and_directory_precedence() {
    let paths = [
        "embeddings/Narrator.safetensors",
        "voices/Narrator.safetensors",
        "voices/sub/other.safetensors",
        "voices/readme.md",
        "default-voice.safetensors",
    ];
    assert_eq!(
        ptts::loader::voice_paths(paths.into_iter().map(str::to_owned)),
        [("Narrator".to_string(), "voices/Narrator.safetensors".to_string())]
    );
}

#[cfg(feature = "hf")]
#[test]
#[ignore = "requires PTTS_TEST_MODEL pointing to a q8 checkpoint"]
fn private_checkpoint_synthesizes_through_the_public_api() {
    let path = std::env::var("PTTS_TEST_MODEL").expect("set PTTS_TEST_MODEL");
    let ck =
        Checkpoint::resolve(path, ResolveOptions { quant: Quant::Q80, weights: None }).unwrap();
    let mut synth = ck
        .builder(ptts::preprocess::Lang::En)
        .device(ptts::synth::DeviceKind::Cpu)
        .build()
        .unwrap();
    ck.register_voices(&mut synth);
    let audio = synth.say("Hello from Phonon.").unwrap();
    assert_eq!(synth.sample_rate(), ck.config.mimi.sample_rate as u32);
    assert!(audio.len() > synth.sample_rate() as usize / 4);
    assert!(audio.iter().all(|x| x.is_finite()));
    assert!(audio.iter().any(|x| x.abs() > 0.01));
}
