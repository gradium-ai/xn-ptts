#pragma once

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace phonon {

// Match only explicitly declared targets. Do not guess compatibility from a
// device name.
inline std::string select_context_binary(const nlohmann::json &binaries,
                                         std::string soc) {
  std::transform(soc.begin(), soc.end(), soc.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (soc.empty())
    throw std::runtime_error("set soc_model to select a compiled QNN model");
  std::string selected;
  for (const auto &item : binaries.items()) {
    const auto &entry = item.value();
    if (!entry.contains("soc_models"))
      continue;
    for (const auto &target : entry.at("soc_models")) {
      if (target != soc)
        continue;
      if (!selected.empty())
        throw std::runtime_error("multiple QNN models match " + soc);
      selected = entry.at("file");
    }
  }
  if (selected.empty())
    throw std::runtime_error("no compiled QNN model for " + soc +
                             "; bundle needs soc_models metadata");
  const std::filesystem::path path(selected);
  if (path.is_absolute())
    throw std::runtime_error("context binary must be relative to the bundle");
  for (const auto &part : path)
    if (part == "..")
      throw std::runtime_error("context binary must stay inside the bundle");
  return selected;
}

} // namespace phonon
