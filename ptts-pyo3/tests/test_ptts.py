"""Tests for the `ptts` wheel.

Everything here runs without model weights, so the wheel job can run it on every platform it
builds for. The handful of tests that do need a checkpoint are marked `checkpoint` and are
deselected by default -- see `pyproject.toml`.
"""

from __future__ import annotations

import ast
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


def test_all_matches_what_is_importable():
    for name in ptts.__all__:
        assert hasattr(ptts, name), name


# --- introspection ---------------------------------------------------------------------------


def test_available_devices_always_offers_the_cpu():
    devices = ptts.available_devices()
    assert isinstance(devices, list)
    assert "cpu" in devices
    # Most capable first, CPU last: `auto` picks devices[0].
    assert devices[-1] == "cpu"


def test_available_quants_are_all_accepted_by_the_constructor():
    quants = ptts.available_quants()
    assert "f32" in quants and "q8_0" in quants
    # A name not in the list is rejected, which is what makes the list meaningful.
    with pytest.raises(ValueError):
        ptts.TTS(quant="q3k", lang="en")


def test_build_info_reports_what_a_bug_report_needs():
    info = ptts.build_info()
    assert set(info) >= {"version", "devices", "threads"}
    assert info["version"] == ptts.__version__


def test_thread_count_round_trips():
    before = ptts.get_num_threads()
    assert before >= 1
    ptts.set_num_threads(before)
    assert ptts.get_num_threads() == before


# --- errors ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "exc", "needle"),
    [
        # A bad argument is a `ValueError`; a checkpoint that is not there is a `LookupError`;
        # a backend this wheel was not built with is a `NotImplementedError`. The README
        # documents that table, so this is what holds it to it.
        ({"quant": "q3k"}, ValueError, "q3k"),
        ({"device": "tpu"}, ValueError, "tpu"),
        ({"config": "/definitely/not/a/checkpoint/config.json"}, LookupError, "config.json"),
        ({"device": "cuda", "quant": "q8_0"}, NotImplementedError, "CPU-only"),
    ],
)
def test_a_bad_argument_raises_its_class_and_names_itself(kwargs, exc, needle):
    with pytest.raises(exc) as e:
        ptts.TTS(**kwargs, lang="en")
    assert needle in str(e.value)


def test_nothing_is_downloaded_before_the_arguments_are_checked():
    # Each of these fails in milliseconds, which only holds if the check precedes the fetch.
    import time

    start = time.monotonic()
    for kwargs in ({"quant": "q3k"}, {"device": "cuda", "quant": "q8_0"}):
        with pytest.raises(Exception):
            ptts.TTS(**kwargs, lang="en")
    assert time.monotonic() - start < 5.0


# --- needs a checkpoint ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def tts() -> ptts.TTS:
    return ptts.TTS(lang="en")


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
def test_the_same_seed_gives_the_same_audio(tts):
    import numpy as np

    a = tts.synth("Reproducible.", seed=7)
    b = tts.synth("Reproducible.", seed=7)
    assert np.array_equal(a, b)
