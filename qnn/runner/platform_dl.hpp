#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <string>
#define RTLD_NOW 0
#define RTLD_LOCAL 0
inline void* dlopen(const char* path, int) {
  int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, nullptr, 0);
  if (!size) return nullptr;
  std::wstring wide(size, L'\0');
  if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide.data(), size)) return nullptr;
  return reinterpret_cast<void*>(LoadLibraryW(wide.c_str()));
}
inline void* dlsym(void* h, const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(h), name));
}
inline const char* dlerror() {
  static thread_local std::string message;
  message = "Windows loader error " + std::to_string(GetLastError());
  return message.c_str();
}
#else
#include <dlfcn.h>
#endif
