// The fp16 storage type: host buffers for float tensors are always fp16.
#pragma once

#if defined(_MSC_VER) && !defined(__clang__)
#error "MSVC has no fp16 type: build with clang (clang-cl, or Visual Studio's ClangCL toolset)"
#endif

namespace phonon {

// GCC only has _Float16 in C++ from version 13; __fp16 is the arm64 storage type
// both GCC and clang know, and converts to and from float.
#if defined(__aarch64__)
using f16 = __fp16;
#else
using f16 = _Float16;
#endif

}  // namespace phonon
