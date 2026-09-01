WASM := target/wasm32-unknown-unknown/release/ptts_web.wasm

.PHONY: build serve clean

build:
	cargo build --release --target wasm32-unknown-unknown
	wasm-bindgen --target web --out-dir pkg $(WASM)
	cp www/index.html www/worker.js www/tokenizer.js pkg/
	@echo "built pkg/ ($$(du -h pkg/ptts_web_bg.wasm | cut -f1) wasm)"

# Serves pkg/ plus the local model directory, so the page can be driven without
# downloading a few hundred MB from HuggingFace on every run.
serve: build
	node server.js

clean:
	rm -rf pkg target
