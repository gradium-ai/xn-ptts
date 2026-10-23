#include "platform_dl.hpp"

#include <iostream>
#include <string>

int main() {
  void* library = dlopen("kernel32.dll", RTLD_NOW | RTLD_LOCAL);
  if (!library || !dlsym(library, "GetCurrentProcessId")) {
    std::cerr << "cannot load a system DLL and resolve its symbol\n";
    return 1;
  }
  if (dlsym(library, "phonon_deliberately_missing_symbol") ||
      std::string(dlerror()).find("Windows loader error") == std::string::npos) {
    return 2;
  }
  if (dlopen("phonon_deliberately_missing_library.dll", RTLD_NOW | RTLD_LOCAL)) {
    return 3;
  }
  // The UTF-8 loader rejects malformed input before calling LoadLibraryW.
  if (dlopen("\xff.dll", RTLD_NOW | RTLD_LOCAL)) {
    return 4;
  }
  std::cout << "Windows loader success and failure paths passed\n";
}
