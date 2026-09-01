WASM := target/wasm32-unknown-unknown/release/ptts_web.wasm

.PHONY: build serve clean

build:
	cargo build --release --target wasm32-unknown-unknown
	wasm-bindgen --target web --out-dir pkg $(WASM)
	cp www/index.html www/worker.js www/tokenizer.js pkg/
	@# A stamp the page can compare against, so a tab left open across a rebuild
	@# says so instead of failing in a way that looks like a code bug.
	@printf '%s %s\n' "$$(git rev-parse --short HEAD 2>/dev/null || echo nogit)" \
	  "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > pkg/build.txt
	@echo "built pkg/ ($$(du -h pkg/ptts_web_bg.wasm | cut -f1) wasm, stamp $$(cat pkg/build.txt))"

# Serves pkg/ plus the local model directory, so the page can be driven without
# downloading a few hundred MB from HuggingFace on every run.
serve: build
	node server.js

clean:
	rm -rf pkg target
