"""Check config requirements through the Hub transport without model downloads."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import ptts


@pytest.fixture
def hub(tmp_path, monkeypatch):
    state = SimpleNamespace(repo="test/phonon", revision="b" * 40, requests=[], denied=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, body):
            path = urlsplit(self.path).path
            state.requests.append(path)
            if "/tree/" in path:
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                if body:
                    self.wfile.write(b"[]")
                return
            self.send_response(401 if state.denied else 404)
            if not state.denied:
                self.send_header("X-Error-Code", "EntryNotFound")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def load(hub):
    model = f"hf://{hub.repo}@{hub.revision}"
    return ptts.TTS(model, device="cpu", quant="q8", lang="en")


def test_phonon_hub_requires_config_before_fetching_weights(hub):
    with pytest.raises(LookupError, match="config.json"):
        load(hub)
    assert not any(p.endswith((".gguf", ".safetensors")) for p in hub.requests)


@pytest.mark.parametrize("repo", ["kyutai/pocket-tts", "kyutai/pocket-tts-without-voice-cloning"])
def test_pocket_repos_also_require_config(hub, repo):
    hub.repo = repo
    with pytest.raises(LookupError, match="config.json"):
        load(hub)


def test_config_authentication_errors_remain_io_errors(hub):
    hub.denied = True
    with pytest.raises(OSError):
        load(hub)
    assert not any(p.endswith((".gguf", ".safetensors")) for p in hub.requests)


def test_the_deprecated_config_still_names_a_hub_repo(hub):
    with pytest.warns(DeprecationWarning), pytest.raises(LookupError, match="config.json"):
        ptts.TTS(config=hub.repo, revision=hub.revision, quant="q8", lang="en")
    assert hub.requests


def test_a_bare_repo_id_is_a_local_path(hub):
    with pytest.raises(LookupError, match=f"hf://{hub.repo}"):
        ptts.TTS(hub.repo, lang="en")
    assert hub.requests == []


def test_without_a_model_the_default_is_fetched(hub):
    repo, revision = ptts.DEFAULT_MODEL.removeprefix("hf://").split("@")
    with pytest.raises(LookupError, match="config.json"):
        ptts.TTS(lang="en", quant="q8")
    assert f"/{repo}/resolve/{revision}/config.json" in hub.requests


def test_without_a_catalog_the_default_is_fetched(hub):
    repo, revision = ptts.DEFAULT_VOICE_CATALOG.removeprefix("hf://").split("@")
    with pytest.raises(LookupError, match="voices.jsonl"):
        ptts.catalog_voices()
    assert hub.requests == [f"/{repo}/resolve/{revision}/voices.jsonl"]


@pytest.mark.parametrize("device", ["cuda", "vulkan", "metal"])
def test_unavailable_device_is_rejected_before_hub_access(hub, device):
    if device in ptts.available_devices():
        pytest.skip("backend is included in this wheel")
    with pytest.raises(NotImplementedError, match=f"device '{device}' is not available") as exc:
        ptts.TTS(f"hf://{hub.repo}", device=device, quant="f32", lang="en")
    assert f"available devices: {', '.join(ptts.available_devices())}" in str(exc.value)
    assert hub.requests == []


def test_quantized_gpu_is_rejected_before_hub_access(hub):
    with pytest.raises(NotImplementedError, match="CPU-only"):
        ptts.TTS(f"hf://{hub.repo}", device="cuda", quant="q8", lang="en")
    assert hub.requests == []
