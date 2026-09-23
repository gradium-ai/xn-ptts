"""Type stubs for the `ptts` extension module.

Hand-written rather than generated: the surface is small, and the docstrings here are what an
editor shows on hover. Keep in step with `ptts-pyo3/src/lib.rs`.
"""

from collections.abc import Iterator
from os import PathLike
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import NDArray

__version__: str

__all__ = [
    "TTS",
    "AudioStream",
    "__version__",
    "available_devices",
    "available_quants",
    "build_info",
    "get_num_threads",
    "set_num_threads",
]

class TTS:
    """A loaded Pocket TTS model.

    `config` is a Hugging Face repo id, a path to a local `config.json`, or `None` for the
    published checkpoint. Weights are downloaded on first use and cached.

    `lang` is required and keyword-only: text is normalized before it is tokenized, and the
    spoken forms of `@`, `+` and `=` differ per language, so there is nothing safe to default
    to. One of `"en"`, `"fr"`, `"de"`, `"es"`, `"pt"`, or `"none"`/`None` to skip normalizing.
    """

    def __init__(
        self,
        config: str | None = None,
        device: str | None = None,
        quant: str | None = None,
        voice: str | None = None,
        temperature: float = 0.5,
        seed: int = 4242424242424242,
        cfg_coef: float | None = None,
        eos_threshold: float | None = None,
        *,
        lang: str | None,
    ) -> None: ...
    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio this model produces, in Hz."""

    @property
    def voices(self) -> list[str]:
        """Names of the registered voices, sorted."""

    @property
    def device(self) -> str:
        """Device the model is running on, e.g. `"cpu"`."""

    @property
    def quant(self) -> str:
        """Weight format actually loaded, e.g. `"q8_0"`."""

    @property
    def voice_prompt_sample_rate(self) -> int:
        """Sample rate `clone_voice` expects its PCM in, in Hz."""

    @property
    def supports_voice_cloning(self) -> bool:
        """True if this checkpoint can clone voices from audio."""

    def synth(
        self,
        text: str,
        *,
        voice: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        cfg_coef: float | None = None,
    ) -> NDArray[np.float32]:
        """Synthesize `text` and return the waveform as a 1-D float32 array."""

    def save(
        self,
        path: str | PathLike[str],
        text: str,
        *,
        voice: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        cfg_coef: float | None = None,
    ) -> float:
        """Synthesize `text` straight to a mono 16-bit WAV file.

        Returns the duration written, in seconds.
        """

    def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        cfg_coef: float | None = None,
    ) -> AudioStream:
        """Synthesize `text`, yielding float32 chunks as the decoder produces them."""

    def add_voice(self, name: str, path: str | PathLike[str]) -> None:
        """Register a voice from a precomputed embedding file."""

    def add_voice_from_embedding(
        self,
        name: str,
        embedding: NDArray[np.float32],
        *,
        null_embedding: NDArray[np.float32] | None = None,
    ) -> None:
        """Register a voice from an in-memory embedding of shape `[T, dim]` or `[1, T, dim]`.

        `null_embedding` is the encoding of equal-length silence, which CFG needs on models
        whose null branch is conditioned on silence.
        """

    def clone_voice(self, name: str, pcm: NDArray[np.float32]) -> None:
        """Clone a voice from ~10s of speech, as float32 PCM at `voice_prompt_sample_rate`.

        Raises `NotImplementedError` when `supports_voice_cloning` is false.
        """

class AudioStream(Iterator[NDArray[np.float32]]):
    """An in-progress generation. Iterate it for float32 chunks.

    Usable as a context manager; leaving the block stops the generation and releases the
    worker threads.
    """

    @property
    def sample_rate(self) -> int:
        """Sample rate of the chunks, in Hz."""

    def close(self) -> None:
        """Stop generating and release the worker threads."""

    def __iter__(self) -> AudioStream: ...
    def __next__(self) -> NDArray[np.float32]: ...
    def __enter__(self) -> AudioStream: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> bool: ...

def get_num_threads() -> int:
    """Number of CPU threads used for tensor ops."""

def set_num_threads(num_threads: int) -> None:
    """Set the number of CPU threads used for tensor ops.

    Call before loading a model: it sizes a global thread pool that is built once.
    """

def available_devices() -> list[str]:
    """The device names this build accepts, most capable first. `"auto"` picks the first."""

def available_quants() -> list[str]:
    """The weight formats `quant=` accepts. All are CPU-only."""

def build_info() -> dict[str, Any]:
    """The runtime's build configuration, for bug reports."""
