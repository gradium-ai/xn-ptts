// SPDX-License-Identifier: MIT OR Apache-2.0
#pragma once
#include <cstdarg>
#include <functional>
#include <string_view>

namespace phonon {
enum class LogLevel { error = 1, warning = 2, info = 3, debug = 4, verbose = 5 };
using LogHandler = std::function<void(LogLevel, std::string_view)>;

// Process-wide embedding sink: QNN's callback has no user-data argument.
// Install before model creation, clear after all models are destroyed. The
// callback receives a borrowed string and must not throw. Calls are made outside
// the handler mutex, so a sink may reconfigure logging without a deadlock.
// With no sink, standalone CLI/environment logging retains its stderr behavior.
void set_qnn_log_handler(LogHandler handler, LogLevel level = LogLevel::warning);
LogLevel qnn_log_level();
void emit_qnn_log(LogLevel level, const char *format, va_list args) noexcept;
} // namespace phonon
