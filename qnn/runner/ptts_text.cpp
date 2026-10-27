#include "ptts_text.hpp"

#include <dlfcn.h>

#include <stdexcept>
#include <memory>

namespace phonon {
namespace {

// The parts of ptts_text.h this file reads; the layout must match it.
struct ptts_text_chunk {
  const char* text;
  const char* prepared;
  const uint32_t* tokens;
  size_t n_tokens;
  uint32_t frames_after_eos;  // after the EOS frame, which is not counted
  uint32_t frame_budget;
  uint32_t seq_budget;
};

struct ptts_text_chunks {
  const ptts_text_chunk* chunks;
  size_t n_chunks;
};

template <typename T>
T symbol(void* lib, const char* name) {
  void* p = dlsym(lib, name);
  if (!p) throw std::runtime_error(std::string("libptts_text has no ") + name);
  return reinterpret_cast<T>(p);
}

}  // namespace

PttsText::PttsText(const std::string& lib, const std::string& tokenizer_json, const std::string& lang) {
  lib_ = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!lib_) throw std::runtime_error("dlopen " + lib + ": " + dlerror());
  auto make = symbol<void* (*)(const char*, const char*, char**)>(lib_, "ptts_text_new");
  split_ = symbol<decltype(split_)>(lib_, "ptts_text_split");
  chunks_free_ = symbol<decltype(chunks_free_)>(lib_, "ptts_text_chunks_free");
  string_free_ = symbol<decltype(string_free_)>(lib_, "ptts_text_string_free");
  free_ = symbol<decltype(free_)>(lib_, "ptts_text_free");
  char* error = nullptr;
  handle_ = make(tokenizer_json.c_str(), lang.c_str(), &error);
  if (!handle_) {
    std::string msg = error ? error : "unknown error";
    if (error) string_free_(error);
    throw std::runtime_error("ptts_text_new: " + msg);
  }
}

PttsText::~PttsText() {
  if (handle_) free_(handle_);
  // Keep the Rust library resident: TLS destructors and worker threads may still use it.
}

std::vector<Chunk> PttsText::split(const std::string& text, size_t max_tokens, double frame_rate) const {
  char* error = nullptr;
  auto* chunks = static_cast<ptts_text_chunks*>(split_(handle_, text.c_str(), max_tokens, frame_rate, &error));
  if (!chunks) {
    std::string msg = error ? error : "unknown error";
    if (error) string_free_(error);
    throw std::runtime_error("ptts_text_split: " + msg);
  }
  auto release = [&](void* value) { chunks_free_(value); };
  std::unique_ptr<void, decltype(release)> guard(chunks, release);
  std::vector<Chunk> out;
  for (size_t i = 0; i < chunks->n_chunks; i++) {
    const ptts_text_chunk& c = chunks->chunks[i];
    Chunk chunk;
    chunk.ids.assign(c.tokens, c.tokens + c.n_tokens);
    chunk.frames_after_eos = int(c.frames_after_eos) + 1;  // ptts plays the EOS frame too
    chunk.frame_budget = int(c.frame_budget);
    out.push_back(std::move(chunk));
  }
  return out;
}

}  // namespace phonon
