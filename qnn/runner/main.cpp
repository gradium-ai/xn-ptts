// phonon: text to a WAV file with a Phonon bundle's graphs on QNN.
//
//   phonon --bundle DIR --text "Hello world." [--voice NAME] [--out out.wav]
//              --lang en|fr|de|es|pt|none [--soc-model SM...] [--backend htp|cpu|gpu] [--lib-dir DIR] [--seed N] [--temperature T]
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "phonon.hpp"

namespace {

void write_wav(const std::string& path, const std::vector<float>& audio, int sample_rate) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
  uint32_t data_bytes = audio.size() * 2;
  f.write("RIFF", 4);
  u32(36 + data_bytes);
  f.write("WAVEfmt ", 8);
  u32(16);
  u16(1);  // PCM
  u16(1);  // mono
  u32(sample_rate);
  u32(sample_rate * 2);
  u16(2);
  u16(16);
  f.write("data", 4);
  u32(data_bytes);
  for (float s : audio) {
    float c = s > 1.f ? 1.f : s < -1.f ? -1.f : s;
    int16_t v = int16_t(c * 32767.f);
    f.write(reinterpret_cast<const char*>(&v), 2);
  }
}

int usage() {
  std::cerr << "usage: phonon --bundle DIR --text TEXT [--voice NAME] [--out FILE.wav]\n"
               "              --lang en|fr|de|es|pt|none [--soc-model SM...] [--backend htp|cpu|gpu] [--lib-dir DIR] [--seed N] [--temperature T]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  phonon::Options opt;
  std::string text, voice, out = "out.wav";
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    try {
      if (a == "--bundle") opt.bundle_dir = next();
      else if (a == "--text") text = next();
      else if (a == "--voice") voice = next();
      else if (a == "--out") out = next();
      else if (a == "--lang") opt.lang = next();
      else if (a == "--soc-model") opt.soc_model = next();
      else if (a == "--backend") opt.backend = next();
      else if (a == "--lib-dir") opt.lib_dir = next();
      else if (a == "--seed") opt.seed = std::stoull(next());
      else if (a == "--temperature") opt.temperature = std::stof(next());
      else return usage();
    } catch (const std::exception& e) {
      std::cerr << e.what() << "\n";
      return usage();
    }
  }
  if (opt.bundle_dir.empty() || text.empty() || opt.lang.empty()) return usage();

  try {
    phonon::Phonon tts(opt);
    phonon::Stats stats;
    std::vector<float> audio = tts.synthesize(text, voice.empty() ? tts.default_voice() : voice, nullptr, &stats);
    write_wav(out, audio, tts.sample_rate());
    double seconds = double(audio.size()) / tts.sample_rate();
    std::printf("%s: %.2f s of audio, %d chunk(s), %d frames\n", out.c_str(), seconds, stats.chunks, stats.frames);
    std::printf("prefill %.1f ms, step %.2f ms/frame, first audio after %.1f ms, total %.1f ms (%.1fx real time)\n",
                stats.prefill_ms / std::max(stats.chunks, 1), stats.step_ms / std::max(stats.frames, 1),
                stats.first_audio_ms, stats.total_ms, seconds * 1000 / stats.total_ms);
    if (stats.step_npu_ms > 0)
      std::printf("on the NPU: prefill %.2f ms/call, step %.2f ms/call (%d calls)\n",
                  stats.prefill_npu_ms / std::max(stats.prefill_calls, 1), stats.step_npu_ms / stats.step_calls,
                  stats.step_calls);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
