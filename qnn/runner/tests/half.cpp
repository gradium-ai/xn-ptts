#include <cmath>
#include <iostream>
#include <limits>

#include "f16.hpp"
int main() {
  int tested = 0;
  for (unsigned i = 0; i < 65536; i++) {
    if (((i >> 10) & 31) == 31) continue;
    phonon::f16 h;
    h.bits = uint16_t(i);
    phonon::f16 back{float(h)};
    if (back.bits != i) {
      std::cerr << "FP16 mismatch at " << i << "\n";
      return 1;
    }
    tested++;
  }
  if (phonon::f16(-0.f).bits != 0x8000 || phonon::f16(65520.f).bits != 0x7c00 ||
      phonon::f16(1.00048828125f).bits != 0x3c00 || phonon::f16(1.00146484375f).bits != 0x3c02 ||
      phonon::f16(std::ldexp(1.f, -25)).bits != 0 ||
      !std::isnan(float(phonon::f16(std::numeric_limits<float>::quiet_NaN())))) {
    std::cerr << "FP16 rounding / special value failure\n";
    return 1;
  }
  std::cout << "FP16 finite round-trip passed: " << tested << " values\n";
}
