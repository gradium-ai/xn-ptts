#!/bin/sh
# The amd64 image's entry point. It carries two builds: x86-64-v3, whose AVX2 and FMA `xn`'s
# fastest kernels use, and x86-64-v2 for CPUs without them, which many small home servers are.
# The v3 build would stop at the first instruction such a CPU lacks.
dir=/usr/local/libexec/ptts
flags=" $(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2) "
for flag in avx avx2 bmi1 bmi2 f16c fma abm movbe xsave; do
    case "$flags" in
        *" $flag "*) ;;
        *)
            echo "This CPU has no $flag; using the x86-64-v2 build, which is slower." >&2
            exec "$dir/ptts-openai-server-x86-64-v2" "$@"
            ;;
    esac
done
exec "$dir/ptts-openai-server-x86-64-v3" "$@"
