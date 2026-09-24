.PHONY: build profiling demo serve test clean

WASM_PACK_ARGS = --target web --no-pack --out-dir pkg/wasm --out-name phonon_tts

# The `phonon-tts` npm package, ready to publish from pkg/.
build:
	rm -rf pkg
	wasm-pack build $(WASM_PACK_ARGS) --release
	node scripts/pack.mjs pkg

# Same, without wasm-opt, keeping names for the browser profiler.
profiling:
	rm -rf pkg
	wasm-pack build $(WASM_PACK_ARGS) --profiling
	node scripts/pack.mjs pkg

# The demo page, using the package exactly as a consumer would, in site/.
demo: build
	rm -rf site
	mkdir -p site
	cp -r pkg site/phonon-tts
	cp www/index.html site/

serve: demo
	python3 -m http.server -d site 8080

# Tests for the JS wrapper that need no browser and no model.
test:
	node --test js/test/*.test.mjs

clean:
	rm -rf pkg site
