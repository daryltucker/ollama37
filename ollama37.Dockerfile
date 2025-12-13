# ===== Stage 1: Build the source code =====
FROM dogkeeper886/ollama37-builder AS builder

# Copy and download dependencies first to leverage Docker cache
COPY go.mod go.sum ./
RUN go mod download

# Copy source code and build
COPY . /usr/local/src/ollama37
WORKDIR /usr/local/src/ollama37
ARG GGML_CUDA_FA=ON
ARG GGML_CUDA_MMQ=ON
RUN CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++ cmake -B build -DOLLAMA_RUNNER_DIR=cuda_v11 \
    -DGGML_CUDA_FA=OFF -DGGML_CUDA_MMQ=OFF \
    && CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++ cmake --build build -j$(nproc) \
    && go build -o ollama .

# ===== Stage 2: Runtime image =====
FROM rockylinux:9-minimal

RUN microdnf install -y ca-certificates \
    && microdnf clean all

# Copy the built binary
COPY --from=builder /usr/local/src/ollama37/ollama /usr/bin/ollama

# Copy the ggml libraries (CPU and CUDA backends)
COPY --from=builder /usr/local/src/ollama37/build/lib/ollama /usr/lib/ollama

# Copy CUDA runtime libraries
COPY --from=builder /usr/local/cuda-11.4/lib64/libcudart.so* /usr/lib/
COPY --from=builder /usr/local/cuda-11.4/lib64/libcublas.so* /usr/lib/
COPY --from=builder /usr/local/cuda-11.4/lib64/libcublasLt.so* /usr/lib/

# Set environment variables
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib:/usr/lib/ollama
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV OLLAMA_HOST=0.0.0.0:11434

# Expose port
EXPOSE 11434

# Set entrypoint and command
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
