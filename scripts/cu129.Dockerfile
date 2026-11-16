# CUDA 12.9 and GCC on a glibc 2.28 baseline.
FROM pytorch/manylinux2_28-builder:cuda12.9@sha256:d26cd7a9489b53310296e86042de630ac947e25a71dd24b547b86ca9a0a3fc79

ARG PYTHON_TAG=cp311-cp311
ENV PATH=/opt/python/${PYTHON_TAG}/bin:/root/.cargo/bin:/usr/local/cuda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
    CARGO_BUILD_JOBS=4 \
    MAX_JOBS=4 \
    NVCC_THREADS=1

RUN yum install -y numactl-devel libibverbs-devel openssl-devel \
    && yum clean all \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh \
    && sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.99.1 \
    && rm /tmp/rustup.sh

RUN python -m pip install --no-cache-dir torch==2.13.0+cu129 \
        --index-url https://download.pytorch.org/whl/cu129 \
    && python -m pip install --no-cache-dir build wheel setuptools-rust \
        'setuptools-scm>=8' 'scikit-build-core>=0.10' 'cmake>=3.31,<4' ninja patchelf

WORKDIR /src
ENTRYPOINT ["python", "scripts/build_cu129.py"]
