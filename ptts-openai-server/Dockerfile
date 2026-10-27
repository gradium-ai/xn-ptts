# syntax=docker/dockerfile:1

# Build from the repository root. Serves the default model, or a model folder or hf:// repo set
# through PTTS_MODEL.
#   docker build -f ptts-openai-server/Dockerfile -t ptts-openai-server .

FROM rust:1-bookworm AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends libopus-dev libmp3lame-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# `.cargo/config.toml` is left out of the build context (see Dockerfile.dockerignore), so nothing
# pins `target-cpu=native`. x86_64 gets x86-64-v3, as the Python wheels do: `xn` picks its
# quantized kernels at compile time, and a lower baseline would cost q8 its AVX kernels. Many
# small home servers have no AVX2, so it also gets an x86-64-v2 build, and `select-cpu.sh`
# starts whichever one the CPU can run.
ARG TARGETARCH
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target,sharing=locked \
    if [ "$TARGETARCH" = "amd64" ]; then \
        for level in v3 v2; do \
            RUSTFLAGS="-C target-cpu=x86-64-$level" cargo build --release --locked \
                -p ptts-openai-server --target-dir "target/x86-64-$level" \
            && install -D "target/x86-64-$level/release/ptts-openai-server" \
                "/out/libexec/ptts/ptts-openai-server-x86-64-$level" || exit 1; \
        done \
        && install -D ptts-openai-server/select-cpu.sh /out/bin/ptts-openai-server; \
    else \
        cargo build --release --locked -p ptts-openai-server \
        && install -D target/release/ptts-openai-server /out/bin/ptts-openai-server; \
    fi

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libmp3lame0 libopus0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 phonon \
    && install -d -o phonon -g phonon /home/phonon/.cache /home/phonon/.cache/huggingface
COPY --from=build /out/ /usr/local/
LABEL org.opencontainers.image.source="https://github.com/gradium-ai/xn-ptts" \
      org.opencontainers.image.description="OpenAI-compatible text-to-speech server for Phonon"
USER phonon
# Every flag has an environment variable (`--help` lists them), so `docker run -e PTTS_LANG=fr`
# or a compose file's `environment:` configures the server. q8 because this image runs on the
# CPU, where it is faster than f32.
ENV HF_HOME=/home/phonon/.cache/huggingface \
    PTTS_LANG=en \
    PTTS_QUANT=q8
# 10200 is the Wyoming protocol, served only when PTTS_WYOMING_ADDR is set.
EXPOSE 8880 10200
ENTRYPOINT ["ptts-openai-server"]
