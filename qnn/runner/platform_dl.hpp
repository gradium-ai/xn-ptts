// dlopen, dlsym and dlerror: <dlfcn.h>, or on Windows the part of it the runner uses.
#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <string>

#define RTLD_NOW 0
#define RTLD_LOCAL 0

// Paths are UTF-8, as everywhere else in the runner.
inline void* dlopen(const char* path, int) {
  int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, nullptr, 0);
  if (!n) return nullptr;
  std::wstring wide(n, L'\0');
  if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide.data(), n)) return nullptr;
  // LoadLibrary wants backslashes, and the runner joins paths with '/'.
  std::replace(wide.begin(), wide.end(), L'/', L'\\');
  return LoadLibraryW(wide.c_str());
}

inline void* dlsym(void* lib, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
}

inline const char* dlerror() {
  static thread_local std::string message;
  message = "Windows error " + std::to_string(GetLastError());
  return message.c_str();
}
#else
#include <dlfcn.h>
#endif
