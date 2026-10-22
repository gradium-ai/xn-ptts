// SPDX-License-Identifier: MIT OR Apache-2.0
#include "qnn_log.hpp"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>

namespace phonon {
namespace {
std::mutex handler_mutex;
LogHandler sink;
LogLevel sink_level = LogLevel::warning;
LogLevel standalone_level() {
  const char *value = std::getenv("PHONON_QNN_LOG");
  const std::string level = value ? value : "";
  if (level == "verbose")
    return LogLevel::verbose;
  if (level == "debug")
    return LogLevel::debug;
  if (level == "info")
    return LogLevel::info;
  return LogLevel::warning;
}
} // namespace
void set_qnn_log_handler(LogHandler handler, LogLevel level) {
  std::lock_guard<std::mutex> lock(handler_mutex);
  sink = std::move(handler);
  sink_level = level;
}
LogLevel qnn_log_level() {
  std::lock_guard<std::mutex> lock(handler_mutex);
  return sink ? sink_level : standalone_level();
}
void emit_qnn_log(LogLevel level, const char *format, va_list args) noexcept {
  try {
    LogHandler callback;
    LogLevel threshold;
    {
      std::lock_guard<std::mutex> lock(handler_mutex);
      callback = sink;
      threshold = callback ? sink_level : standalone_level();
    }
    if (int(level) > int(threshold) || !format)
      return;
    va_list copy;
    va_copy(copy, args);
    const int length = std::vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    if (length < 0)
      return;
    std::string message(size_t(length), '\0');
    va_copy(copy, args);
    std::vsnprintf(message.data(), message.size() + 1, format, copy);
    va_end(copy);
    if (callback)
      callback(level, message);
    else
      std::fprintf(stderr, "[qnn] %s\n", message.c_str());
  } catch (...) {
    // Exceptions must never cross the vendor C callback boundary.
  }
}
} // namespace phonon
