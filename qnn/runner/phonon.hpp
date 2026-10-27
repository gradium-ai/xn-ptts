// Phonon on QNN: the generation loop around a bundle's prefill and step graphs.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "f16.hpp"
#include "ptts_text.hpp"
#include "qnn_model.hpp"

namespace phonon {

struct Options {
  std::string bundle_dir;
  std::string backend = "htp";  // htp (context binary on the NPU), cpu or gpu (DLCs)
  std::string lib_dir;          // where libQnn*.so live; empty: the loader's search path
  std::string lang;             // explicit normalization language, or "none"
  std::string soc_model;        // Android Build.SOC_MODEL, matched against bundle soc_models
  std::string text_library;     // packaged native library; never execute code from a model download
  uint64_t seed = 0;
  float temperature = -1;  // < 0: the bundle's
};

struct Stats {
  bool cancelled = false;
  size_t samples = 0;
  int chunks = 0;
  int frames = 0;
  double prefill_ms = 0;
  double step_ms = 0;
  double first_audio_ms = -1;
  double total_ms = 0;
  // With PHONON_QNN_PROFILE=1 on the HTP: time on the NPU, summed over the calls.
  int prefill_calls = 0, step_calls = 0;
  double prefill_npu_ms = 0, step_npu_ms = 0;
};

class Phonon {
 public:
  explicit Phonon(const Options& options);

  std::vector<std::string> voices() const;
  // "default" when the bundle has it, else its first voice.
  std::string default_voice() const;
  int sample_rate() const { return sample_rate_; }

  // Calls on_audio with each 80 ms frame as it is generated. Returns the whole utterance.
  std::vector<float> synthesize(const std::string& text, const std::string& voice,
                                const std::function<bool(const float*, size_t)>& on_audio = nullptr,
                                Stats* stats = nullptr,
                                const std::function<bool()>& should_stop = nullptr,
                                bool collect_audio = true);

 private:
  struct Voice {
    std::string name;
    std::vector<f16> kv;  // [2L, H, n, D]
    int length = 0;
  };

  void generate_chunk(const Chunk& chunk, const Voice& voice,
                      const std::function<bool(const float*, size_t)>& on_audio, std::vector<float>& out,
                      Stats& stats, std::chrono::steady_clock::time_point start,
                      const std::function<bool()>& should_stop, bool collect_audio);

  // Every graph buffer, allocated once from the model (shared with the NPU on the
  // HTP) and reused. What step feeds back to itself has two copies that swap.
  struct Buffers {
    f16* kv = nullptr;  // the flow LM cache, [2L, H, slots, D]
    int32_t* tokens = nullptr;
    int32_t* pos = nullptr;
    int32_t* frame = nullptr;
    f16* prefill_kv = nullptr;  // prefill's new keys and values
    f16* emb[2] = {nullptr, nullptr};
    f16* noise = nullptr;
    f16* audio = nullptr;
    f16* eos = nullptr;
    f16* step_kv = nullptr;  // step's new keys and values
    f16* latent = nullptr;
    std::vector<f16*> states[2];
    size_t kv_count = 0, emb_count = 0, noise_count = 0, audio_count = 0;
    std::vector<size_t> state_counts;
  };
  void alloc_buffers();
  Buffers buf_;

  std::unique_ptr<QnnModel> model_;
  std::unique_ptr<PttsText> text_;
  std::vector<Voice> voices_;
  std::vector<f16> bos_;
  std::mt19937_64 rng_;

  int sample_rate_ = 24000;
  double frame_rate_ = 12.5;
  int cache_slots_ = 0, prefill_tokens_ = 0, max_tokens_ = 50;
  int layers2_ = 0, heads_ = 0, head_dim_ = 0;
  float noise_std_ = 0, eos_threshold_ = 0;
  int min_frames_before_eos_ = 0;
};

}  // namespace phonon
