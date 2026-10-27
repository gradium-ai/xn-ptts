#include "phonon.hpp"
#include "bundle.hpp"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <map>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace phonon {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

template <typename T>
std::vector<T> read_raw(const std::string& path, size_t expected_count) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  size_t bytes = f.tellg();
  if (bytes != expected_count * sizeof(T))
    throw std::runtime_error(path + ": " + std::to_string(bytes) + " bytes, expected " +
                             std::to_string(expected_count * sizeof(T)));
  std::vector<T> v(expected_count);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), bytes);
  return v;
}

std::string lib(const Options& o, const std::string& name) {
  return o.lib_dir.empty() ? name : o.lib_dir + "/" + name;
}

}  // namespace

Phonon::Phonon(const Options& o) : rng_(o.seed) {
  if (o.lang.empty()) throw std::runtime_error("set an explicit normalization language (or none)");
  const std::string dir = o.bundle_dir + "/";
  std::ifstream mf(dir + "metadata.json");
  if (!mf) throw std::runtime_error("no metadata.json in " + o.bundle_dir);
  nlohmann::json meta = nlohmann::json::parse(mf);

  const auto& gen = meta.at("generation");
  sample_rate_ = gen.at("sample_rate");
  frame_rate_ = gen.at("frame_rate");
  cache_slots_ = gen.at("cache_slots");
  prefill_tokens_ = gen.at("prefill_tokens");
  max_tokens_ = gen.at("max_tokens_per_chunk");
  float temperature = o.temperature >= 0 ? o.temperature : float(gen.at("temperature"));
  noise_std_ = std::sqrt(temperature);
  eos_threshold_ = gen.at("eos_threshold");
  min_frames_before_eos_ = gen.at("min_frames_before_eos");
  const auto& layout = gen.at("layout");
  layers2_ = 2 * int(layout.at("layers"));
  heads_ = layout.at("heads");
  head_dim_ = layout.at("head_dim");

  const auto& bos = gen.at("bos");
  size_t bos_count = 1;
  for (int d : bos.at("shape")) bos_count *= d;
  bos_ = read_raw<f16>(dir + std::string(bos.at("file")), bos_count);

  for (const auto& v : meta.at("voices")) {
    Voice voice;
    voice.name = v.at("name");
    voice.length = v.at("length");
    voice.kv = read_raw<f16>(dir + std::string(v.at("file")), size_t(layers2_) * heads_ * voice.length * head_dim_);
    voices_.push_back(std::move(voice));
  }

  const auto& text = meta.at("text");
  std::string library = o.text_library;
  if (library.empty()) {
    // The standalone CLI can load the text frontend from its supplied bundle.
#if defined(__ANDROID__)
    const char* platform = "android-arm64";
#else
    const char* platform = "linux-arm64";
#endif
    const auto& entry = text.contains("library") ? text.at("library") : nlohmann::json("libptts_text.so");
    std::string name = entry.is_object() ? std::string(entry.at(platform)) : std::string(entry);
    std::ifstream in_bundle(dir + name);
    library = in_bundle ? dir + name : lib(o, name);
  }
  text_ = std::make_unique<PttsText>(library, dir + std::string(text.at("tokenizer")), o.lang);

  const auto& rt = meta.at("runtime");
  std::vector<std::pair<std::string, std::string>> files;
  std::string backend_lib;
  if (o.backend == "htp") {
    backend_lib = "libQnnHtp.so";
    files.push_back({"", dir + std::string(select_context_binary(rt.at("context_binaries"), o.soc_model))});
  } else if (o.backend == "cpu" || o.backend == "gpu") {
    backend_lib = o.backend == "cpu" ? "libQnnCpu.so" : "libQnnGpu.so";
    for (const auto& [graph, file] : rt.at("dlcs").items()) files.push_back({graph, dir + std::string(file)});
  } else {
    throw std::runtime_error("unknown backend " + o.backend + " (htp, cpu or gpu)");
  }
  model_ = std::make_unique<QnnModel>(lib(o, backend_lib), lib(o, "libQnnSystem.so"), files);
  alloc_buffers();
}

void Phonon::alloc_buffers() {
  const Graph& prefill = model_->graph("prefill");
  const Graph& step = model_->graph("step");
  auto f = [&](const TensorInfo& t) { return static_cast<f16*>(model_->alloc(t)); };
  auto i = [&](const TensorInfo& t) { return static_cast<int32_t*>(model_->alloc(t)); };
  buf_.kv = f(prefill.input("kv_cache"));  // step's kv_cache has the same shape
  buf_.kv_count = prefill.input("kv_cache").count;
  buf_.tokens = i(prefill.input("tokens"));
  buf_.pos = i(prefill.input("pos"));
  buf_.frame = i(step.input("frame"));
  buf_.prefill_kv = f(prefill.output("output_0"));
  // Step's outputs, in order: audio, EOS logit, next emb, new keys/values, latent, Mimi states.
  buf_.emb[0] = f(step.input("emb"));
  buf_.emb[1] = f(step.output("output_2"));
  buf_.emb_count = step.input("emb").count;
  buf_.noise = f(step.input("noise"));
  buf_.noise_count = step.input("noise").count;
  buf_.audio = f(step.output("output_0"));
  buf_.audio_count = step.output("output_0").count;
  buf_.eos = f(step.output("output_1"));
  buf_.step_kv = f(step.output("output_3"));
  buf_.latent = f(step.output("output_4"));
  for (int k = 0;; k++) {
    std::string name = "mimi_states_" + std::to_string(k);
    bool found = false;
    for (const auto& t : step.inputs) found |= t.name == name;
    if (!found) break;
    buf_.states[0].push_back(f(step.input(name)));
    buf_.states[1].push_back(f(step.output("output_" + std::to_string(5 + k))));
    buf_.state_counts.push_back(step.input(name).count);
  }
  if (bos_.size() != buf_.emb_count) throw std::runtime_error("bos.bin does not match step's emb input");
}

std::vector<std::string> Phonon::voices() const {
  std::vector<std::string> names;
  for (const auto& v : voices_) names.push_back(v.name);
  return names;
}

std::string Phonon::default_voice() const {
  for (const auto& v : voices_)
    if (v.name == "default") return v.name;
  if (voices_.empty()) throw std::runtime_error("the bundle has no voices");
  return voices_.front().name;
}

std::vector<float> Phonon::synthesize(const std::string& text, const std::string& voice_name,
                                         const std::function<bool(const float*, size_t)>& on_audio,
                                         Stats* stats_out, const std::function<bool()>& should_stop, bool collect_audio) {
  const Voice* voice = nullptr;
  for (const auto& v : voices_)
    if (v.name == voice_name) voice = &v;
  if (!voice) throw std::runtime_error("no voice " + voice_name);

  Stats stats;
  std::vector<float> out;
  auto t0 = Clock::now();
  model_->set_performance_mode(true);
  struct ResetPerformance {
    QnnModel* model;
    ~ResetPerformance() { model->set_performance_mode(false); }
  } reset{model_.get()};
  for (const auto& chunk : text_->split(text, max_tokens_, frame_rate_)) {
    if (should_stop && should_stop()) { stats.cancelled = true; break; }
    generate_chunk(chunk, *voice, on_audio, out, stats, t0, should_stop, collect_audio);
    stats.chunks++;
    if (stats.cancelled) break;
  }
  stats.total_ms = ms_since(t0);
  if (model_->profiling()) {
    for (const auto& [graph, t] : model_->timing()) {
      if (graph == "prefill") stats.prefill_calls = t.calls, stats.prefill_npu_ms = t.npu_us / 1000;
      if (graph == "step") stats.step_calls = t.calls, stats.step_npu_ms = t.npu_us / 1000;
    }
  }
  if (stats_out) *stats_out = stats;
  return out;
}

void Phonon::generate_chunk(const Chunk& chunk, const Voice& voice,
                               const std::function<bool(const float*, size_t)>& on_audio, std::vector<float>& out,
                               Stats& stats, Clock::time_point start, const std::function<bool()>& should_stop, bool collect_audio) {
  const std::vector<int>& ids = chunk.ids;
  const int n = ids.size();
  if (n == 0) return;
  // prefill runs in pieces of prefill_tokens_, each needing that many free slots.
  const int pieces = (n + prefill_tokens_ - 1) / prefill_tokens_;
  if (voice.length + (pieces - 1) * prefill_tokens_ + prefill_tokens_ > cache_slots_)
    throw std::runtime_error("text chunk too long for the cache (" + std::to_string(n) + " tokens)");
  const int frames_after_eos = chunk.frames_after_eos;
  const int max_frames = std::min(chunk.frame_budget, cache_slots_ - voice.length - n);

  // The cache starts as the voice prompt, in slots 0..length-1. Each chunk starts
  // from the voice again, as ptts does.
  const size_t row = size_t(head_dim_);
  const size_t plane = size_t(cache_slots_) * row;  // one (layer, k/v, head)
  f16* kv = buf_.kv;
  std::fill(kv, kv + buf_.kv_count, f16(0));
  for (int lh = 0; lh < layers2_ * heads_; lh++)
    std::memcpy(&kv[lh * plane], &voice.kv[size_t(lh) * voice.length * row], voice.length * row * sizeof(f16));
  int pos = voice.length;

  // Writes a [2L, H, s, D] block's first `count` positions into the cache at `at`.
  auto write_kv = [&](const f16* block, int s, int count, int at) {
    for (int lh = 0; lh < layers2_ * heads_; lh++)
      std::memcpy(&kv[lh * plane + size_t(at) * row], &block[size_t(lh) * s * row], count * row * sizeof(f16));
  };

  const Graph& prefill = model_->graph("prefill");
  const Graph& step = model_->graph("step");

  {
    auto t = Clock::now();
    for (int first = 0; first < n; first += prefill_tokens_) {
      if (should_stop && should_stop()) { stats.cancelled = true; return; }
      const int count = std::min(prefill_tokens_, n - first);
      std::fill(buf_.tokens, buf_.tokens + prefill_tokens_, 0);
      std::copy(ids.begin() + first, ids.begin() + first + count, buf_.tokens);
      *buf_.pos = pos;
      model_->execute(prefill, {{"tokens", buf_.tokens}, {"kv_cache", kv}, {"pos", buf_.pos}},
                      {{"output_0", buf_.prefill_kv}});
      write_kv(buf_.prefill_kv, prefill_tokens_, count, pos);
      pos += count;
    }
    stats.prefill_ms += ms_since(t);
  }

  // Step reads emb and the Mimi states from set `cur` and writes the next ones
  // into the other set; the two swap after every frame.
  int cur = 0;
  std::copy(bos_.begin(), bos_.end(), buf_.emb[0]);
  for (size_t k = 0; k < buf_.states[0].size(); k++)
    std::fill(buf_.states[0][k], buf_.states[0][k] + buf_.state_counts[k], f16(0));
  std::normal_distribution<float> normal(0.0f, noise_std_);

  int eos_frame = -1;
  std::vector<float> pcm(buf_.audio_count);
  for (int frame = 0; frame < max_frames; frame++) {
    if (should_stop && should_stop()) { stats.cancelled = true; return; }
    auto t = Clock::now();
    for (size_t k = 0; k < buf_.noise_count; k++) buf_.noise[k] = f16(normal(rng_));
    *buf_.pos = pos;
    *buf_.frame = frame;
    std::map<std::string, const void*> in = {{"emb", buf_.emb[cur]}, {"noise", buf_.noise}, {"kv_cache", kv},
                                             {"pos", buf_.pos},       {"frame", buf_.frame}};
    std::map<std::string, void*> outs = {{"output_0", buf_.audio},       {"output_1", buf_.eos},
                                         {"output_2", buf_.emb[1 - cur]}, {"output_3", buf_.step_kv},
                                         {"output_4", buf_.latent}};
    for (size_t k = 0; k < buf_.states[0].size(); k++) {
      in["mimi_states_" + std::to_string(k)] = buf_.states[cur][k];
      outs["output_" + std::to_string(5 + k)] = buf_.states[1 - cur][k];
    }
    model_->execute(step, in, outs);
    stats.step_ms += ms_since(t);
    write_kv(buf_.step_kv, 1, 1, pos);
    pos++;
    cur = 1 - cur;

    if (eos_frame < 0 && frame >= min_frames_before_eos_ && float(*buf_.eos) > eos_threshold_) eos_frame = frame;
    if (eos_frame >= 0 && frame >= eos_frame + frames_after_eos) break;

    for (size_t k = 0; k < buf_.audio_count; k++) pcm[k] = float(buf_.audio[k]);
    if (collect_audio) out.insert(out.end(), pcm.begin(), pcm.end());
    stats.samples += pcm.size();
    stats.frames++;
    if (stats.first_audio_ms < 0) stats.first_audio_ms = ms_since(start);
    if (on_audio && !on_audio(pcm.data(), pcm.size())) { stats.cancelled = true; return; }
  }
}

}  // namespace phonon
