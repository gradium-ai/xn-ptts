"""Phonon: text to 24 kHz speech, on device.

    import ptts

    tts = ptts.TTS(lang="en")  # the default model, downloaded on first use
    tts.save("out.wav", "Hello world")

No PyTorch, no `espeak-ng`, no system packages -- the whole runtime is a Rust
extension in this wheel. See https://github.com/gradium-ai/xn-ptts.
"""

from ._ptts import (
    DEFAULT_MODEL,
    DEFAULT_VOICES,
    TTS,
    AudioStream,
    __version__,
    available_devices,
    available_quants,
    build_info,
    catalog_voices,
    get_num_threads,
    set_num_threads,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_VOICES",
    "TTS",
    "AudioStream",
    "__version__",
    "available_devices",
    "available_quants",
    "build_info",
    "catalog_voices",
    "get_num_threads",
    "set_num_threads",
]
