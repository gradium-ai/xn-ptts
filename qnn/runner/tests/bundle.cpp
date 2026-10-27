#include "bundle.hpp"
#include <iostream>
#include <stdexcept>

int main() {
  using nlohmann::json;
  const json binaries = {
      {"S25", {{"file", "s25.bin"}, {"soc_models", {"SM8750"}}}},
      {"S24", {{"file", "s24.bin"}, {"soc_models", {"SM8650"}}}}};
  if (phonon::select_context_binary(binaries, "SM8650") != "s24.bin")
    return 1;
  if (phonon::select_context_binary(binaries, "sm8750") != "s25.bin")
    return 6;
  auto rejects = [](const json &entries, const std::string &soc) {
    try {
      phonon::select_context_binary(entries, soc);
      return false;
    } catch (const std::exception &) {
      return true;
    }
  };
  if (!rejects(binaries, "") || !rejects(binaries, "unsupported"))
    return 2;
  auto duplicate = binaries;
  duplicate["other"] = binaries["S25"];
  if (!rejects(duplicate, "SM8750"))
    return 3;
  auto legacy = binaries;
  legacy["S25"].erase("soc_models");
  if (!rejects(legacy, "SM8750"))
    return 4;
  for (auto file : {"../outside.bin", "/outside.bin"}) {
    auto invalid = binaries;
    invalid["S25"]["file"] = file;
    if (!rejects(invalid, "SM8750"))
      return 5;
  }
  std::cout << "QNN bundle compatibility checks passed\n";
}
