FROM rust:slim-bookworm
WORKDIR /llmserver-rs
COPY . /llmserver-rs
RUN apt update
RUN apt install clang curl libssl-dev pkg-config cmake libsentencepiece-dev libsentencepiece0 -y
RUN curl -L https://github.com/airockchip/rknn-llm/raw/refs/heads/main/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so -o /lib/librkllmrt.so
RUN curl -L https://github.com/airockchip/rknn-toolkit2/raw/refs/heads/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so -o /lib/librknnrt.so
RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app

RUN apt update && apt install libssl3 libgomp1 ca-certificates libsentencepiece0 -y
COPY --from=0 /lib/librkllmrt.so /lib/librkllmrt.so
COPY --from=0 /lib/librknnrt.so /lib/librknnrt.so
COPY --from=0 /llmserver-rs/target/release/llmserver-rs /app/llmserver-rs
COPY ./assets /app/assets
EXPOSE 8443
ENTRYPOINT ["/app/llmserver-rs"]