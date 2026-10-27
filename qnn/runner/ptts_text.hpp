// ptts's own text front end (normalization, chunking, tokenization), loaded at run
// time from libptts_text: the token ids are exactly the ones the Rust model uses.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace phonon {

// One chunk of text to generate, as token ids plus how far generation may go.
struct Chunk {
  std::vector<int> ids;
  int frames_after_eos = 0;  // frames played from the EOS frame on, that one included
  int frame_budget = 0;      // most frames the chunk may generate
};

class PttsText {
 public:
  // lib: path to libptts_text.so; lang: "en", "fr", ... or "none".
  PttsText(const std::string& lib, const std::string& tokenizer_json, const std::string& lang);
  ~PttsText();
  PttsText(const PttsText&) = delete;
  PttsText& operator=(const PttsText&) = delete;

  std::vector<Chunk> split(const std::string& text, size_t max_tokens, double frame_rate) const;

 private:
  void* lib_ = nullptr;
  void* handle_ = nullptr;
  void* (*split_)(const void*, const char*, size_t, double, char**) = nullptr;
  void (*chunks_free_)(void*) = nullptr;
  void (*string_free_)(char*) = nullptr;
  void (*free_)(void*) = nullptr;
};

}  // namespace phonon
