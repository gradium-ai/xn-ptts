// The fade-in at the start of each chunk, as ptts::plan::FadeIn does it.
#pragma once

#include <cmath>
#include <cstddef>

namespace phonon {

// Samples each chunk fades in over, at `sample_rate`: 20 ms.
inline size_t fade_in_len(int sample_rate) { return size_t(sample_rate) * 20 / 1000; }

// Mimi starts each chunk from a blank state, and the first few milliseconds it decodes from
// it hold a short burst, a click in the silence before each sentence. Fades in `pcm`, the
// chunk's `n` samples from sample `offset` on, over its first `len` samples.
inline void fade_in(float* pcm, size_t n, size_t offset, size_t len) {
  for (size_t i = 0; i < n && offset + i < len; i++)
    pcm[i] *= 0.5f - 0.5f * std::cos(3.14159265358979f * float(offset + i) / float(len));
}

}  // namespace phonon
