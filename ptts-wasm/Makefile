.PHONY: build profiling demo serve test clean check-wasm-opt check-model-dir check-threads-toolchain threads-toolchain

WASM_PACK_ARGS = --target web --no-pack --out-dir pkg/wasm --out-name phonon_tts
# Both builds can run on the GPU: the package uses WebGPU when the browser has it, and the
# CPU, on one thread or several, when it does not.
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

# The `phonon-tts` npm package, ready to publish from pkg/: both builds, and the JS around them.
build: check-wasm-opt check-threads-toolchain
	rm -rf pkg
	wasm-pack build $(WASM_PACK_ARGS) --release -- --features $(FEATURES)
	$(THREADS_WASM_PACK)
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

# The demo page, using the package exactly as a consumer would, in site/. MODEL_DIR is the
# model folder to serve beside it; the page loads whatever weights and voices it holds.
demo: check-model-dir build
	rm -rf site
	mkdir -p site
	cp -r pkg site/phonon-tts
	cp www/index.html site/
	ln -s "$(abspath $(MODEL_DIR))" site/model
	node scripts/demo-model.mjs site/model > site/model.json

# Cross-origin isolated, so the demo runs on the threaded build.
serve: demo
	node scripts/serve.mjs site 8080

# Checked before the build, so a missing folder fails in a second rather than after it.
check-model-dir:
	@if [ -z "$(MODEL_DIR)" ] || [ ! -f "$(MODEL_DIR)/tokenizer.json" ]; then \
	  echo "make demo and make serve need MODEL_DIR, a model folder holding tokenizer.json, the"; \
	  echo "weights and its voices: make serve MODEL_DIR=/path/to/model"; \
	  exit 1; \
	fi

# Tests for the JS wrapper that need no browser and no model. Needs Node 22.7 or
# later, for module-syntax detection on the `.js` modules under js/.
test:
	node --test js/test/*.test.mjs

clean:
	rm -rf pkg site
