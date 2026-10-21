#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "phonon.hpp"

namespace {
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
static void wav(const std::string& file, const std::vector<float>& audio, int sr) {
  std::ofstream f(file, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write WAV");
  auto u32 = [&](uint32_t v) { f.write((char*)&v, 4); };
  auto u16 = [&](uint16_t v) { f.write((char*)&v, 2); };
  f.write("RIFF", 4);
  u32(36 + audio.size() * 2);
  f.write("WAVEfmt ", 8);
  u32(16);
  u16(1);
  u16(1);
  u32(sr);
  u32(sr * 2);
  u16(2);
  u16(16);
  f.write("data", 4);
  u32(audio.size() * 2);
  for (float x : audio) {
    if (!std::isfinite(x)) throw std::runtime_error("nonfinite PCM");
    int16_t y = int16_t(std::max(-1.f, std::min(1.f, x)) * 32767);
    f.write((char*)&y, 2);
  }
  if (!f) throw std::runtime_error("WAV write failed");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    phonon::Options o;
    o.backend = "htp";
    int repeats = 5;
    std::string text, out, voice;
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i];
      if (i + 1 >= argc) throw std::runtime_error("missing argument");
      std::string v = argv[++i];
      if (a == "--bundle")
        o.bundle_dir = v;
      else if (a == "--lib-dir")
        o.lib_dir = v;
      else if (a == "--text-library")
        o.text_library = v;
      else if (a == "--soc-model")
        o.soc_model = v;
      else if (a == "--text")
        text = v;
      else if (a == "--lang")
        o.lang = v;
      else if (a == "--out")
        out = v;
      else if (a == "--repeats")
        repeats = std::stoi(v);
      else if (a == "--voice")
        voice = v;
      else
        throw std::runtime_error("unknown argument: " + a);
    }
    if (repeats < 1 || repeats > 100 || out.empty() || text.empty())
      throw std::runtime_error("provide text, out and 1..100 repeats");
#ifdef _WIN32
    _putenv_s("PHONON_QNN_PROFILE", "1");
#else
    setenv("PHONON_QNN_PROFILE", "1", 1);
#endif
    auto t = Clock::now();
    phonon::Phonon p(o);
    double load = ms(t);
    if (voice.empty()) voice = p.default_voice();
    nlohmann::json report = {{"backend", "QNN HTP"},
                             {"cpu_model_fallback", false},
                             {"gpu_model_fallback", false},
                             {"load_ms", load},
                             {"voice", voice},
                             {"runs", nlohmann::json::array()}};
    double prev_prefill = 0, prev_step = 0;
    int prev_pc = 0, prev_sc = 0;
    for (int i = 0; i < repeats; i++) {
      phonon::Stats s;
      t = Clock::now();
      double callback = -1;
      auto audio = p.synthesize(
          text, voice,
          [&](const float*, size_t) {
            if (callback < 0) callback = ms(t);
            return true;
          },
          &s);
      double np = s.prefill_npu_ms - prev_prefill, ns = s.step_npu_ms - prev_step;
      int pc = s.prefill_calls - prev_pc, sc = s.step_calls - prev_sc;
      if (np <= 0 || ns <= 0 || pc < 1 || sc < 1)
        throw std::runtime_error("strict NPU proof failed: both graphs require positive accelerator timings");
      prev_prefill = s.prefill_npu_ms;
      prev_step = s.step_npu_ms;
      prev_pc = s.prefill_calls;
      prev_sc = s.step_calls;
      double secs = double(audio.size()) / p.sample_rate(), energy = 0;
      for (float v : audio) energy += v * v;
      if (audio.empty() || energy < 1e-10) throw std::runtime_error("empty/silent output");
      std::string file = out + "-" + std::to_string(i) + ".wav";
      wav(file, audio, p.sample_rate());
      nlohmann::json row = {{"run", i},
                            {"warm", i > 0},
                            {"first_pcm_callback_ms", callback},
                            {"generation_ms", s.total_ms},
                            {"audio_seconds", secs},
                            {"rtf", s.total_ms / (secs * 1000)},
                            {"prefill_npu_ms", np},
                            {"step_npu_ms", ns},
                            {"prefill_calls", pc},
                            {"step_calls", sc},
                            {"wav", file}};
      report["runs"].push_back(row);
      std::cout << row.dump() << std::endl;
    }
    std::ofstream f(out + ".json");
    if (!f) throw std::runtime_error("cannot write JSON report");
    f << report.dump(2);
    if (!f) throw std::runtime_error("JSON report write failed");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
