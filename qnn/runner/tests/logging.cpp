// SPDX-License-Identifier: MIT OR Apache-2.0
#include "qnn_log.hpp"
#include <stdexcept>
#include <string>

void emit(phonon::LogLevel level, const char *format, ...) {
  va_list args;
  va_start(args, format);
  phonon::emit_qnn_log(level, format, args);
  va_end(args);
}
int main() {
  using namespace phonon;
  std::string received;
  set_qnn_log_handler(
      [&](LogLevel level, std::string_view value) {
        if (level != LogLevel::info)
          throw std::runtime_error("level");
        received = value;
        set_qnn_log_handler([](LogLevel, std::string_view) { throw std::runtime_error("sink"); });
      },
      LogLevel::info);
  if (qnn_log_level() != LogLevel::info)
    return 1;
  emit(LogLevel::debug, "filtered");
  if (!received.empty())
    return 2;
  emit(LogLevel::info, "model=%s frames=%d", "phonon", 3);
  if (received != "model=phonon frames=3")
    return 3;
  emit(LogLevel::warning, "sink exception must be contained");
  set_qnn_log_handler({});
  return 0;
}
