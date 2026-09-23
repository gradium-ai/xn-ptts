# xn-ptts

Pocket TTS: text to 24 kHz speech, on device, in Rust. No PyTorch, no `espeak-ng`, no system
packages.

Try the wasm version online on [github.io](https://laurentmazare.github.io/pocket-tts).

## Python

```bash
uvx ptts --lang en "Hello world" -o out.wav   # nothing to install
pip install ptts                      # or keep it around
```

```python
import ptts

ptts.TTS(lang="en").save("out.wav", "Hello world")
```

See [`ptts-pyo3/README.md`](ptts-pyo3/README.md) for the rest of the API, and
[on PyPI](https://pypi.org/project/ptts/).

## Rust

```bash
cargo run --release --example say --features hf -- "hello world"
```

`ptts/` is the library, `ptts-pyo3/` the Python bindings, `ptts-wasm/` the browser build and
`ptts-ws-server/` a streaming websocket server.
