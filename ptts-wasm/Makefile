.PHONY: build profiling demo serve test clean check-wasm-opt check-cargo-about check-model-dir check-threads-toolchain threads-toolchain

WASM_PACK_ARGS = --target web --no-pack --out-dir pkg/wasm --out-name phonon_tts
# Both builds can run on the GPU: the package uses WebGPU when asked to, and otherwise the CPU,
# on one thread or several.
FEATURES = webgpu

# The threaded build, in pkg/wasm-threads/. Wasm threads share memory, which needs std itself
# rebuilt with atomics, so this one uses a pinned nightly with rust-src. Shared memory also
# has to be imported with a declared maximum, and wasm-bindgen needs the heap base and the
# TLS symbols exported to set each Web Worker up; `+atomics` implies none of that. RUSTFLAGS
# replaces every rustflags setting in .cargo/config.toml, so the SIMD flags and the getrandom
# backend are repeated here. Its own target directory keeps it from rebuilding the
# single-threaded one each time.
THREADS_TOOLCHAIN = nightly-2026-09-08
THREADS_RUSTFLAGS = -C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128,+relaxed-simd \
	-C link-arg=--import-memory -C link-arg=--shared-memory -C link-arg=--max-memory=2147483648 \
	-C link-arg=--export=__heap_base -C link-arg=--export=__wasm_init_tls \
	-C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base \
	--cfg getrandom_backend="wasm_js"
THREADS_WASM_PACK = RUSTUP_TOOLCHAIN=$(THREADS_TOOLCHAIN) RUSTFLAGS='$(THREADS_RUSTFLAGS)' \
	CARGO_TARGET_DIR=../target/wasm-threads \
	wasm-pack build --target web --no-pack --out-dir pkg/wasm-threads --out-name phonon_tts --release \
	-- --features threads,$(FEATURES) -Z build-std=std,panic_abort

# Both builds again without Relaxed SIMD, for browsers that lack it: Safari, and so every
# browser on iOS. RUSTFLAGS replaces the config's flags, which is what drops it. pack.mjs
# keeps only their .wasm, beside the Relaxed one, as phonon_tts_simd_bg.wasm.
SIMD_RUSTFLAGS = -C target-feature=+simd128 --cfg getrandom_backend="wasm_js"
SIMD_WASM_PACK = RUSTFLAGS='$(SIMD_RUSTFLAGS)' CARGO_TARGET_DIR=../target/wasm-simd \
	wasm-pack build --target web --no-pack --out-dir pkg/wasm-simd --out-name phonon_tts --release \
	-- --features $(FEATURES)
THREADS_SIMD_WASM_PACK = RUSTUP_TOOLCHAIN=$(THREADS_TOOLCHAIN) \
	RUSTFLAGS='$(subst +relaxed-simd,-relaxed-simd,$(THREADS_RUSTFLAGS))' \
	CARGO_TARGET_DIR=../target/wasm-threads-simd \
	wasm-pack build --target web --no-pack --out-dir pkg/wasm-threads-simd --out-name phonon_tts --release \
	-- --features threads,$(FEATURES) -Z build-std=std,panic_abort

# The licence notices of every crate compiled into the modules, which the package ships beside
# them: the threaded build's features cover the single-threaded one's.
LICENSES = cargo about generate --locked --fail --features threads,$(FEATURES) \
	-c about.toml about.hbs -o pkg/THIRD_PARTY_LICENSES.md

# The `phonon-tts` npm package, ready to publish from pkg/: both builds, each with and without
# Relaxed SIMD, the JS around them and the third-party licence notices.
build: check-wasm-opt check-threads-toolchain check-cargo-about
	rm -rf pkg
	wasm-pack build $(WASM_PACK_ARGS) --release -- --features $(FEATURES)
	$(THREADS_WASM_PACK)
	$(SIMD_WASM_PACK)
	$(THREADS_SIMD_WASM_PACK)
	$(LICENSES)
	node scripts/pack.mjs pkg

# Same, without wasm-opt, keeping names for the browser profiler. Single-threaded build only.
profiling:
	rm -rf pkg
	wasm-pack build $(WASM_PACK_ARGS) --profiling -- --features $(FEATURES)
	node scripts/pack.mjs pkg

# wasm-pack runs the first wasm-opt on PATH, and downloads binaryen 117 when there is none.
# Releases up to 123 abort on this module ("UNREACHABLE executed" in their interpreter), so
# the release build needs binaryen 124 or later on PATH. CI builds with exactly 124.
WASM_OPT_MIN = 124
check-wasm-opt:
	@v=$$(wasm-opt --version 2>/dev/null | sed -n 's/^wasm-opt version \([0-9]*\).*/\1/p'); \
	if [ -z "$$v" ] || [ "$$v" -lt $(WASM_OPT_MIN) ]; then \
	  echo "make build needs binaryen's wasm-opt $(WASM_OPT_MIN) or later on PATH (found: $${v:-none})."; \
	  echo "Install it with 'brew install binaryen', or from https://github.com/WebAssembly/binaryen/releases."; \
	  echo "For an unoptimized local build without binaryen, run 'make profiling' instead."; \
	  exit 1; \
	fi

check-cargo-about:
	@cargo about --version >/dev/null 2>&1 || { \
	  echo "make build needs cargo-about, to write the package's third-party licence notices."; \
	  echo "Install it with 'cargo install cargo-about --locked --features cli'."; \
	  exit 1; }

check-threads-toolchain:
	@rustup run $(THREADS_TOOLCHAIN) rustc --print sysroot >/dev/null 2>&1 && \
	  [ -d "$$(rustup run $(THREADS_TOOLCHAIN) rustc --print sysroot)/lib/rustlib/src/rust/library" ] || { \
	  echo "make build needs the $(THREADS_TOOLCHAIN) toolchain with rust-src, for the threaded build."; \
	  echo "Install it with 'make threads-toolchain'."; \
	  exit 1; }

# Installs the toolchain the threaded build is pinned to. CI runs this too, so the version
# lives in one place.
threads-toolchain:
	rustup toolchain install $(THREADS_TOOLCHAIN) --profile minimal --component rust-src --target wasm32-unknown-unknown

# The demo page, using the package exactly as a consumer would, in site/. MODEL_DIR, if set,
# is a model folder to serve beside it, and the page loads whatever weights and voices it holds;
# without it, the page loads the package's default model from the Hugging Face Hub.
demo: check-model-dir build
	rm -rf site
	mkdir -p site
	cp -r pkg site/phonon-tts
	cp www/index.html site/
	if [ -n "$(MODEL_DIR)" ]; then \
	  ln -s "$(abspath $(MODEL_DIR))" site/model && \
	  node scripts/demo-model.mjs site/model > site/model.json; \
	fi

# Cross-origin isolated, so the demo runs on the threaded build.
serve: demo
	node scripts/serve.mjs site 8080

# Checked before the build, so a missing folder fails in a second rather than after it.
check-model-dir:
	@if [ -n "$(MODEL_DIR)" ] && [ ! -f "$(MODEL_DIR)/tokenizer.json" ]; then \
	  echo "MODEL_DIR must be a model folder holding tokenizer.json, the weights and its"; \
	  echo "voices, or unset for the default model: make serve MODEL_DIR=/path/to/model"; \
	  exit 1; \
	fi

# Tests for the JS wrapper that need no browser and no model. Needs Node 22.7 or
# later, for module-syntax detection on the `.js` modules under js/.
test:
	node --test js/test/*.test.mjs

clean:
	rm -rf pkg site
