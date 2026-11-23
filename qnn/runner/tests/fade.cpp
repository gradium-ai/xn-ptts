#include "fade.hpp"
#include <iostream>
#include <vector>

int main() {
  const size_t len = phonon::fade_in_len(24000);
  if (len != 480) return 1;
  std::vector<float> whole(len + 100, 1.0f);
  phonon::fade_in(whole.data(), whole.size(), 0, len);
  if (whole[0] != 0.0f || whole[len - 1] < 0.99f) return 2;
  for (size_t i = 1; i < len; i++)
    if (whole[i] < whole[i - 1]) return 3;
  for (size_t i = len; i < whole.size(); i++)
    if (whole[i] != 1.0f) return 4;
  // In pieces, carrying on from the samples already faded.
  std::vector<float> pieces(len + 100, 1.0f);
  phonon::fade_in(pieces.data(), 7, 0, len);
  phonon::fade_in(pieces.data() + 7, pieces.size() - 7, 7, len);
  if (pieces != whole) return 5;
  std::cout << "fade-in checks passed\n";
}
