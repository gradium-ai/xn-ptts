// Test only: LD_PRELOAD shim that makes QNN's CPU backend start off a Qualcomm SoC.
//
// The backend reads /sys/devices/soc0/soc_id (or the older /sys/devices/system/soc/soc0/id)
// and fails backendCreate with QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED when it is missing,
// as in a Linux container on a Mac. Those paths are redirected to $FAKE_SOC_ID_FILE.
//
//   gcc -shared -fPIC -o fake_soc.so fake_soc.c -ldl
//   FAKE_SOC_ID_FILE=/tmp/soc_id LD_PRELOAD=./fake_soc.so phonon ...
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char* redirect(const char* path) {
  if (path && (strcmp(path, "/sys/devices/soc0/soc_id") == 0 || strcmp(path, "/sys/devices/system/soc/soc0/id") == 0)) {
    const char* fake = getenv("FAKE_SOC_ID_FILE");
    if (fake) return fake;
  }
  return path;
}

#define REAL(name) static __typeof__(name)* real_##name; if (!real_##name) real_##name = dlsym(RTLD_NEXT, #name)

int access(const char* path, int mode) {
  REAL(access);
  return real_access(redirect(path), mode);
}

int faccessat(int dirfd, const char* path, int mode, int flags) {
  REAL(faccessat);
  return real_faccessat(dirfd, redirect(path), mode, flags);
}

int open(const char* path, int flags, ...) {
  REAL(open);
  va_list ap;
  va_start(ap, flags);
  int mode = va_arg(ap, int);
  va_end(ap);
  return real_open(redirect(path), flags, mode);
}

int open64(const char* path, int flags, ...) {
  REAL(open64);
  va_list ap;
  va_start(ap, flags);
  int mode = va_arg(ap, int);
  va_end(ap);
  return real_open64(redirect(path), flags, mode);
}

int openat(int dirfd, const char* path, int flags, ...) {
  REAL(openat);
  va_list ap;
  va_start(ap, flags);
  int mode = va_arg(ap, int);
  va_end(ap);
  return real_openat(dirfd, redirect(path), flags, mode);
}

FILE* fopen(const char* path, const char* mode) {
  REAL(fopen);
  return real_fopen(redirect(path), mode);
}

FILE* fopen64(const char* path, const char* mode) {
  REAL(fopen64);
  return real_fopen64(redirect(path), mode);
}
