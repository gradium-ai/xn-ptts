"""Tests for the `ptts` wheel.

Everything here runs without model weights, so the wheel job can run it on every platform it
builds for. The handful of tests that do need a checkpoint are marked `checkpoint` and are
deselected by default -- see `pyproject.toml`.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from importlib import metadata
from pathlib import Path

import pytest

import ptts


# --- packaging -------------------------------------------------------------------------------


def test_version_is_a_real_version():
    assert isinstance(ptts.__version__, str)
    assert ptts.__version__.count(".") >= 2, ptts.__version__


def test_the_package_is_marked_typed():
    # PEP 561: without this file, type checkers ignore the package entirely.
    assert (Path(ptts.__file__).parent / "py.typed").is_file()


def test_the_stubs_ship_with_the_wheel():
    assert (Path(ptts.__file__).parent / "__init__.pyi").is_file()


@pytest.mark.parametrize(
    ("name", "needle"),
    [("LICENSE-MIT", "Permission is hereby granted"), ("LICENSE-APACHE", "Apache License")],
)
def test_the_licenses_ship_with_the_wheel(name, needle):
    # `ptts-pyo3/LICENSE-*` are symlinks to the repository root. A checkout without symlink
    # support makes each one a file holding its target's path, which would then ship instead.
    dist = metadata.distribution("ptts")
    assert dist.metadata["License-Expression"] == "MIT OR Apache-2.0"
    text = dist.read_text(f"licenses/{name}")
    assert text is not None and needle in text, f"{name} in the wheel holds {text!r:.80}"


def test_the_stubs_cover_everything_the_extension_exports():
    # The stubs are hand-written, so this is what catches them drifting from the Rust.
    stub = Path(ptts.__file__).parent / "__init__.pyi"
    tree = ast.parse(stub.read_text())
    declared = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    declared |= {
        target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        for target in [node.target]
    }
    missing = set(ptts.__all__) - declared
    assert not missing, f"exported but not in the stubs: {sorted(missing)}"


def test_the_stubs_all_matches_the_package_all():
    # Type checkers read `__all__` from the stub, not from `__init__.py`.
    tree = ast.parse((Path(ptts.__file__).parent / "__init__.pyi").read_text())
    (stub_all,) = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    ]
    assert sorted(stub_all) == sorted(ptts.__all__)


def _stub_parameters(fn: ast.FunctionDef) -> list[tuple[str, str, object]]:
    args = fn.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    params = [
        (arg.arg, "POSITIONAL_ONLY" if i < len(args.posonlyargs) else "POSITIONAL_OR_KEYWORD", d)
        for i, (arg, d) in enumerate(zip(positional, defaults))
    ]
    if args.vararg:
        params.append((args.vararg.arg, "VAR_POSITIONAL", None))
    params += [(arg.arg, "KEYWORD_ONLY", d) for arg, d in zip(args.kwonlyargs, args.kw_defaults)]
    if args.kwarg:
        params.append((args.kwarg.arg, "VAR_KEYWORD", None))
    return [
        (name, kind, inspect.Parameter.empty if d is None else ast.literal_eval(d))
        for name, kind, d in params
        if name != "self"
    ]


def _stub_callables():
    # Plain functions and methods, plus `__init__` as the class's own signature. Properties
    # and the other dunders are left out: their signatures are fixed by Python, not by us.
    # Names are resolved in the test, so one the extension lacks fails that case alone
    # rather than the collection of this whole file.
    tree = ast.parse((Path(ptts.__file__).parent / "__init__.pyi").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            yield pytest.param(node, node.name, id=node.name)
        elif isinstance(node, ast.ClassDef):
            for fn in node.body:
                if not isinstance(fn, ast.FunctionDef) or any(
                    isinstance(d, ast.Name) and d.id == "property" for d in fn.decorator_list
                ):
                    continue
                if fn.name == "__init__":
                    yield pytest.param(fn, node.name, id=node.name)
                elif not fn.name.startswith("__"):
                    path = f"{node.name}.{fn.name}"
                    yield pytest.param(fn, path, id=path)


@pytest.mark.parametrize(("fn", "path"), list(_stub_callables()))
def test_the_stub_signatures_match_the_extension(fn, path):
    # The check above only compares names. This one compares parameters, their kinds and
    # their defaults, which is where the stubs drifted before: `temperature` said 0.5 while
    # the extension used 0.3, and `rewrites` was missing.
    obj = ptts
    for name in path.split("."):
        obj = getattr(obj, name)
    if isinstance(obj, type) and (sys.version_info < (3, 10) or sys.implementation.name == "pypy"):
        # Neither an abi3 build on 3.9 nor PyPy gives a class its signature. The wheel jobs
        # that test on a later CPython still check the constructor.
        pytest.skip("classes carry no signature on Python 3.9 or PyPy")
    runtime = [
        (p.name, p.kind.name, p.default)
        for p in inspect.signature(obj).parameters.values()
        if p.name != "self"
    ]
    assert _stub_parameters(fn) == runtime


def test_all_matches_what_is_importable():
    for name in ptts.__all__:
        assert hasattr(ptts, name), name


def test_all_covers_everything_the_extension_exports():
    # The other half of the drift check: the test above compares the stubs against `__all__`,
    # so an `m.add_*` in the Rust that `__init__.py` never re-exports is invisible to both it
    # and to `import ptts`. `dir(_ptts)` is the ground truth.
    from ptts import _ptts

    exported = {name for name in dir(_ptts) if not name.startswith("_")} | {"__version__"}
    missing = exported - set(ptts.__all__)
    assert not missing, f"in the extension but not re-exported: {sorted(missing)}"


# --- introspection ---------------------------------------------------------------------------


def test_available_devices_always_offers_the_cpu():
    devices = ptts.available_devices()
    assert isinstance(devices, list)
    assert "cpu" in devices
    # Most capable first, CPU last: `auto` picks devices[0].
    assert devices[-1] == "cpu"


def test_available_quants_lists_the_known_formats_and_rejects_others():
    quants = ptts.available_quants()
    assert "f32" in quants and "q8_0" in quants
    # A name not in the list is rejected, which is what makes the list meaningful.
    with pytest.raises(ValueError):
        ptts.TTS(quant="q3k", lang="en")


def test_build_info_reports_what_a_bug_report_needs():
    info = ptts.build_info()
    assert set(info) >= {"version", "devices", "threads"}
    assert info["version"] == ptts.__version__


def test_normalize_reads_text_as_tts_does():
    assert ptts.normalize("It costs $5.", lang="en") == "It costs 5 dollars."
    assert ptts.normalize("Ça coûte 250 €.", lang="fr") == "Ça coûte 250 euros."
    assert ptts.normalize("Call 3:45PM.", lang="en", rewrites="all") == "Call 3-45 PM."
    assert ptts.normalize("It costs $5.", lang="en", rewrites="none") == "It costs $5."
    assert ptts.normalize("It costs $5.", lang="none") == "It costs $5."
    assert ptts.normalize("It costs $5.", lang=None) == "It costs $5."


@pytest.mark.parametrize(
    ("kwargs", "needle"), [({"lang": "xx"}, "xx"), ({"lang": "en", "rewrites": "colours"}, "colours")]
)
def test_normalize_rejects_what_tts_rejects(kwargs, needle):
    with pytest.raises(ValueError, match=needle):
        ptts.normalize("hello", **kwargs)
    with pytest.raises(TypeError):
        ptts.normalize("hello")


def test_thread_count_round_trips():
    before = ptts.get_num_threads()
    assert before >= 1
    ptts.set_num_threads(before)
    assert ptts.get_num_threads() == before


# --- voice catalog ---------------------------------------------------------------------------


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    (tmp_path / "voices.jsonl").write_text(
        '{"filename": "voices/a1.safetensors", "name": "Kauã", "voice_id": "a1", "language": "pt"}\n'
        '{"filename": "voices/b2.safetensors", "name": "Zoey", "locale": "en-us"}\n',
        encoding="utf-8",
    )
    return tmp_path


def test_catalog_voices_lists_a_local_catalog(catalog):
    for source in (catalog, catalog / "voices.jsonl", str(catalog)):
        voices = ptts.catalog_voices(source)
        assert [v["name"] for v in voices] == ["Kauã", "Zoey"]
        assert voices[0] == {"name": "Kauã", "voice_id": "a1", "language": "pt", "description": None}


def test_a_missing_catalog_is_a_lookup_error(tmp_path):
    with pytest.raises(LookupError):
        ptts.catalog_voices(tmp_path / "missing.jsonl")


def test_list_voices_needs_no_lang_or_model(catalog, capsys):
    from ptts.__main__ import main

    assert main(["--list-voices", "--voice-catalog", str(catalog)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["Kauã\tpt\ta1\t-", "Zoey\t-\t-\t-"]


def test_the_default_voice_catalog_is_pinned_to_a_commit():
    repo, revision = ptts.DEFAULT_VOICE_CATALOG.removeprefix("hf://").split("@")
    assert repo.count("/") == 1
    assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)


# --- errors ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "exc", "needle"),
    [
        # A bad argument is a `ValueError`; a checkpoint that is not there is a `LookupError`;
        # a backend this wheel was not built with is a `NotImplementedError`. `to_py_err` in
        # `src/lib.rs` is where that mapping lives, and this is what holds it to it.
        ({"quant": "q3k"}, ValueError, "q3k"),
        ({"device": "tpu"}, ValueError, "tpu"),
        ({"model": "./missing", "revision": "test-commit"}, ValueError, "revision"),
        ({"config": "./missing", "revision": "test-commit"}, ValueError, "revision"),
        ({"model": "/definitely/not/a/checkpoint/config.json"}, LookupError, "config.json"),
        ({"model": "owner/name"}, LookupError, "hf://owner/name"),
        ({"model": "hf://owner"}, ValueError, "hf://owner"),
        ({"device": "cuda", "quant": "q8_0"}, NotImplementedError, "CPU-only"),
    ],
)
@pytest.mark.filterwarnings("ignore:config= and revision= are deprecated:DeprecationWarning")
def test_a_bad_argument_raises_its_class_and_names_itself(kwargs, exc, needle):
    with pytest.raises(exc) as e:
        ptts.TTS(**kwargs, lang="en")
    assert needle in str(e.value)


def test_nothing_is_downloaded_before_the_arguments_are_checked():
    # Each of these fails in milliseconds, which only holds if the check precedes the fetch.
    # The bound is deliberately loose: this is a smoke test for the ordering, not a benchmark,
    # and a cold runner's import-time page-ins should not be able to fail it. Fetching a
    # checkpoint and building a model is well over it even on a fast runner.
    import time

    start = time.monotonic()
    for kwargs in ({"quant": "q3k"}, {"device": "cuda", "quant": "q8_0"}):
        with pytest.raises(Exception):
            ptts.TTS(**kwargs, lang="en")
    assert time.monotonic() - start < 30.0


# --- needs a checkpoint ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def tts() -> ptts.TTS:
    # Private rehearsals use the installed wheel and a local directory or hf:// repo.
    # Model tests require an explicit checkpoint; ordinary wheel checks need none.
    model = os.environ.get("PTTS_TEST_MODEL")
    if not model:
        pytest.skip("set PTTS_TEST_MODEL for checkpoint synthesis tests")
    return ptts.TTS(
        model,
        lang="en",
        device="cpu",
        quant="q8",
    )


@pytest.mark.checkpoint
def test_synth_returns_float32_audio(tts):
    import numpy as np

    pcm = tts.synth("Hello world.")
    assert pcm.dtype == np.float32
    assert pcm.ndim == 1
    assert len(pcm) > tts.sample_rate // 4, "suspiciously short"
    assert abs(pcm).max() > 0.01, "silence"


@pytest.mark.checkpoint
def test_save_writes_a_playable_wav(tmp_path, tts):
    import wave

    out = tmp_path / "out.wav"
    seconds = tts.save(out, "Hello world.")
    with wave.open(str(out)) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == tts.sample_rate
        assert w.getnframes() / w.getframerate() == pytest.approx(seconds, abs=0.01)


@pytest.mark.checkpoint
def test_stream_yields_chunks_and_closes(tts):
    with tts.stream("Hello world.") as audio:
        assert audio.sample_rate == tts.sample_rate
        chunks = [next(audio), next(audio)]
    assert all(len(c) for c in chunks)


@pytest.mark.checkpoint
def test_an_unknown_voice_lists_the_ones_that_exist(tts):
    with pytest.raises(LookupError) as e:
        tts.synth("hi", voice="definitely-not-a-voice")
    assert tts.voices[0] in str(e.value)


@pytest.mark.checkpoint
def test_an_unknown_voice_with_an_unreadable_catalog_still_names_the_voices(tts, tmp_path):
    model = os.environ["PTTS_TEST_MODEL"]
    missing = tmp_path / "voices.jsonl"
    other = ptts.TTS(model, lang="en", device="cpu", quant="q8", voice_catalog=missing)
    with pytest.raises(LookupError) as e:
        other.synth("hi", voice="definitely-not-a-voice")
    message = str(e.value)
    assert "definitely-not-a-voice" in message and other.voices[0] in message
    assert str(missing) in message


@pytest.mark.checkpoint
def test_a_language_the_model_was_not_trained_for_is_warned_about(tts):
    other = next((lang for lang in ("en", "fr", "de", "es", "pt") if lang not in tts.languages), None)
    if not tts.languages or other is None:
        pytest.skip("the checkpoint lists no languages, or all of them")
    with pytest.warns(UserWarning, match=f"normalized as {other}"):
        ptts.TTS(os.environ["PTTS_TEST_MODEL"], lang=other, device="cpu", quant="q8")


@pytest.mark.checkpoint
def test_the_same_seed_gives_the_same_audio(tts):
    import numpy as np

    a = tts.synth("Reproducible.", seed=7)
    b = tts.synth("Reproducible.", seed=7)
    # The decoder drains available latents in batches, whose sizes depend on scheduling.
    # Different GEMM shapes can change rounding even for identical sampled latents.
    # This absolute tolerance is below one 16-bit PCM step; duration must still match.
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
    c = tts.synth("Reproducible.", seed=8)
    assert a.shape != c.shape or not np.allclose(a, c, rtol=0, atol=1e-6)
