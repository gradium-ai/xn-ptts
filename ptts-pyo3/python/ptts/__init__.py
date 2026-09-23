"""Pocket TTS: text to 24 kHz speech, on device.

    import ptts

    tts = ptts.TTS(lang="en")
    tts.save("out.wav", "Hello world")

No PyTorch, no `espeak-ng`, no system packages -- the whole runtime is a Rust
extension in this wheel. See https://github.com/gradium-ai/xn-ptts.
"""

from ._ptts import (
    TTS,
    AudioStream,
    __version__,
    available_devices,
    available_quants,
    build_info,
    get_num_threads,
    set_num_threads,
)

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
