// The fp16 storage type: host buffers for float tensors are always fp16.
#pragma once

#ifdef _MSC_VER
#include <cstdint>
#include <cstring>
namespace phonon {
struct f16 {
  uint16_t bits = 0;
  f16() = default;
  explicit f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, m = x & 0x7fffff;
    int e = int((x >> 23) & 255) - 127 + 15;
    if (e >= 31) {
      bits = uint16_t(sign | 0x7c00 | (m && ((x >> 23) & 255) == 255 ? 0x200 : 0));
      return;
    }
    if (e <= 0) {
      if (e < -10) {
        bits = uint16_t(sign);
        return;
      }
      m |= 0x800000;
      int shift = 14 - e;
      uint32_t rounded = (m + ((1u << (shift - 1)) - 1) + ((m >> shift) & 1)) >> shift;
      bits = uint16_t(sign | rounded);
      return;
    }
    m += 0xfff + ((m >> 13) & 1);
    if (m & 0x800000) {
      m = 0;
      ++e;
    }
    bits = uint16_t(sign | (e >= 31 ? 0x7c00 : (uint32_t(e) << 10) | (m >> 13)));
  }
  operator float() const {
    uint32_t sign = uint32_t(bits & 0x8000) << 16, m = bits & 1023, e = (bits >> 10) & 31, x;
    if (!e) {
      if (!m)
        x = sign;
      else {
        int exp = -14;
        while (!(m & 1024)) {
          m <<= 1;
          --exp;
        }
        x = sign | (uint32_t(exp + 127) << 23) | ((m & 1023) << 13);
      }
    } else if (e == 31)
      x = sign | 0x7f800000 | (m << 13);
    else
      x = sign | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
  }
};
static_assert(sizeof(f16) == 2);
}  // namespace phonon

#else
namespace phonon {

// GCC only has _Float16 in C++ from version 13; __fp16 is the arm64 storage type
// both GCC and clang know, and converts to and from float.
#if defined(__aarch64__)
using f16 = __fp16;
#else
using f16 = _Float16;
#endif

}  // namespace phonon

#endif
