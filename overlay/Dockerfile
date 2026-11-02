# Reuse the official CUDA artifacts and rebuild the Rust frontend from the
# same source as the Python overlay. See overlay/README.md.
ARG BASE_IMAGE
FROM rust:1.95-bookworm AS rust-build
WORKDIR /build
COPY rust/ rust/
COPY rust-toolchain.toml ./
ARG VLLM_RS_BUILD_VERSION=dev
ENV VLLM_RS_BUILD_VERSION=${VLLM_RS_BUILD_VERSION}
ENV CARGO_BUILD_JOBS=4
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/rust/target \
    cargo build --manifest-path rust/Cargo.toml --locked --release \
        --package vllm-cmd --bin vllm-rs --features native-tls-vendored \
    && cp rust/target/release/vllm-rs /vllm-rs

FROM ${BASE_IMAGE}
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY --from=rust-build /vllm-rs /usr/local/lib/python3.12/dist-packages/vllm/vllm-rs
