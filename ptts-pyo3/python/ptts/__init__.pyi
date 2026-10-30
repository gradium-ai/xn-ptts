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
DEFAULT_MODEL: str
"""The model `TTS` loads when none is given, as `hf://owner/name@revision`."""
DEFAULT_VOICE_CATALOG: str
"""The voice catalog `TTS` and `catalog_voices` use when none is given, as
`hf://owner/name@revision`."""

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_VOICE_CATALOG",
    "TTS",
    "AudioStream",
    "__version__",
    "available_devices",
    "available_quants",
    "build_info",
    "catalog_voices",
    "get_num_threads",
    "normalize",
    "set_num_threads",
]

class TTS:
    """A loaded Phonon model.

    `model` must name a local directory or config.json, or a Hugging Face repo as
    `"hf://owner/name"` or `"hf://owner/name@revision"`. Left out, it is `DEFAULT_MODEL`. Local
    paths never download anything; Hub files are downloaded on first use and cached. The revision, a
    branch, tag, or commit, applies to every Hub file. `config` and `revision` are the
    deprecated older spelling, where a bare `"owner/name"` named a repo.

    `lang` is required and keyword-only: text is normalized before it is tokenized, and the
    spoken forms of `@`, `+` and `=` differ per language, so there is nothing safe to default
    to. One of `"en"`, `"fr"`, `"de"`, `"es"`, `"pt"`, `"ar"`, `"hi"`, `"ko"`, `"pl"`, `"sv"`,
    `"fi"`, `"no"`, `"it"`, `"nl"`, `"ja"`, `"zh"`, `"da"`, `"tr"`, or `"none"`/`None` to skip
    normalizing.

    `rewrites` picks which word rewrites run on the normalized text: `"default"` (what `None`
    means: numbers, currency, dashed-digits, emails, urls, abbreviations, elongations), `"all"`
    (those and phones, times, dates), `"none"`, or a comma-separated list of rule names. The
    rules only read `"en"`, `"fr"`, `"de"`, `"es"` and `"pt"`.

    `voice_catalog` is where a voice the checkpoint does not have is looked up: a Hugging Face
    repo as `"hf://owner/name[@revision]"`, or a local `voices.jsonl` or folder holding one.
    Left out, it is `DEFAULT_VOICE_CATALOG`. A voice is matched by name, in any case and with or
    without accents, or by its catalog `voice_id`, and only the voices used are downloaded.
    """

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        quant: str | None = None,
        voice: str | None = None,
        temperature: float = 0.3,
        seed: int = 4242424242424242,
        cfg_coef: float | None = None,
        eos_threshold: float | None = None,
        *,
        lang: str | None,
        rewrites: str | None = None,
        conditions: dict[str, str | float] | None = None,
        voice_catalog: str | PathLike[str] | None = None,
        config: str | None = None,
        revision: str | None = None,
    ) -> None: ...
    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio this model produces, in Hz."""

    @property
    def voices(self) -> list[str]:
        """Names of the registered voices, sorted. Catalog voices join them once used."""

    @property
    def languages(self) -> list[str]:
        """The languages the model was trained for, as listed in its config."""

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
        """Synthesize `text`, yielding float32 chunks as the decoder produces them.

        Generation pauses once its bounded buffers fill if the stream is unread, and
        resumes as chunks are consumed. Closing or dropping it stops its workers.
        """

    def add_voice(self, name: str, path: str | PathLike[str]) -> None:
        """Register a voice from a precomputed embedding file."""

    def add_voice_catalog(self, path: str | PathLike[str]) -> None:
        """Register every voice of a JSONL catalog, one `{"name", "filename"}` object per line
        with `filename` relative to the catalog."""

    def set_conditions(self, conditions: dict[str, str | float]) -> None:
        """Replace the conditions, without reloading the model.

        Values go through `str()`, as `conditions=` does, and an invalid one leaves the
        previous conditions in place.
        """

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

    Unread streams pause generation once their bounded buffers fill. Consuming chunks
    lets generation resume. Leaving a context manager block stops generation and waits
    for the worker threads to finish.
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
        # `*_args` on the Rust side: positional only, so `__exit__(exc_type=None)` is a
        # `TypeError` at runtime and the stub should not type-check it.
        /,
    ) -> bool: ...

def get_num_threads() -> int:
    """Number of CPU threads used for tensor ops."""

def set_num_threads(num_threads: int) -> None:
    """Set the number of CPU threads used for tensor ops.

    Call before loading a model: it sizes a global thread pool that is built once.
    """

def available_devices() -> list[str]:
    """The device names this build accepts, most capable first. `"auto"` picks the first,
    and the CPU if that one fails to start."""

def available_quants() -> list[str]:
    """The weight formats `quant=` accepts. All are CPU-only."""

def build_info() -> dict[str, Any]:
    """The runtime's build configuration, for bug reports."""

def catalog_voices(catalog: str | PathLike[str] | None = None) -> list[dict[str, str | None]]:
    """The voices of a catalog, as dicts of `name`, `voice_id`, `language` and `description`.

    `catalog` is as for `TTS`'s `voice_catalog`, `DEFAULT_VOICE_CATALOG` when left out. Only the catalog is downloaded, no voice and
    no model.
    """

def normalize(text: str, *, lang: str | None, rewrites: str | None = None) -> str:
    """`text` normalized as `TTS` with the same `lang` and `rewrites` normalizes it before
    tokenizing, without loading a model.

    Accented letters a checkpoint's tokenizer lacks are folded afterwards, by `TTS`, and are kept
    here.
    """
