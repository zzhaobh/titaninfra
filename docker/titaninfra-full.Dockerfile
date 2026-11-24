# TitanInfra full GPU training environment
#
# Complete upstream runtime definition: CUDA 13.2.1 + cuDNN, Python 3.12,
# PyTorch 2.15, DeepEP, FlashMLA, vLLM, MosaicML Streaming, multimodal
# libraries, profiling tools, and runtime verification helpers.
#
# BuildKit is recommended. Project sources and model/data assets are mounted
# at runtime rather than embedded in the image.

FROM <previous tag>` file. The base's steps are kept verbatim
# and in order; the ex-layers live in PART 8, right before the inherited PART 7
# (whole-image self-check + final cleanup) so that self-check covers them too.
#
# Current forward profile (researched 2026-11-06; not yet built or runtime-verified)
# ---------------------------------------------------------------------------------
# Ubuntu 24.04 / Python 3.12 / CUDA toolkit 13.2.1, with official stable wheels:
#   TORCH_VERSION=2.15.0        TORCHVISION_VERSION=0.29.2
#   TORCHAUDIO_VERSION=2.11.1   TRITON_EXPECTED_PREFIX=3.9
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
# Torch 2.15 wheel metadata pins NCCL 2.30.7, NVSHMEM 3.7.2 and cuDNN 9.26.0.51.
# The base stays at 13.2.1 because NVIDIA publishes no 13.2.2 devel image; the
# torch wheel's pip cuda-toolkit is 13.2.2. CUDA 13.x requires a 580+ driver.
#
# vLLM defaults to the latest stable release, v0.33.0, source-built against the
# installed torch 2.15. Upstream v0.33.0 declares torch 2.14.0, so this is a forward
# compatibility build, not an upstream-supported torch combination. The source step
# applies an inline vLLM compatibility patch ported from PR #62519: NCCL2 lazy
# process groups plus the cuBLAS pre-profile warmup. No C++/CUDA source is patched.
# The build-local VLLM_MAIN_CUDA_VERSION=13.0 selects the published cu130 Rust
# frontend; C++/CUDA extensions are compiled locally.
#
# Build from an empty directory containing only this Dockerfile:
#   docker buildx build --platform linux/amd64 --load \
#     -f titaninfra_env.dockerfile -t titaninfra-env:local .
# Required before promotion: whole-image imports, GPU generate, TP2/PP2 NCCL2 smoke,
# tier-0, kernel parity and the RL battery (including case_rl_05 and case_rl_32).
#
# Historical profiles remain evidence, not current build instructions:
# - torch 2.14.1 + vLLM 0.28.0 formed titaninfra-env:2026-10-07-stable.
# - torch 2.14.1 + vLLM 0.31.0 formed titaninfra-env:2026-10-08-vllm31; its
#   case_rl_05 regression is recorded in docs/tutorials/troubleshooting.md.
#   The historical experiment tag titaninfra-env:2026-10-08-vllm31 must not be
#   inferred to describe the defaults below. See image_tags.json for both receipts.
#
# 2026-11-18 dependency re-audit against the torchtitan tip that is mounted at
# runtime. Authoritative source read at torchtitan HEAD ea6334273 (upstream base
# 361ae5df6): .ci/docker/requirements.txt. Because torchtitan and titaninfra are
# both installed with `pip install --no-deps`, every import they make at runtime
# must already exist in this image; a version that is merely "close" fails as a
# ModuleNotFoundError/AttributeError on the GPU node, not at build time.
# Changed by that audit (each with its evidence inline):
#   spmd_types 0.2.5 -> 0.3.2     (0.2.5 has no spmd_types.rules submodule)
#   renderers 0.1.11 -> 0.1.12    (upstream pin for the same source tree)
#   attn-gym 0.0.17 -> 0.0.20     (attn_gym.mla exists only from 0.0.20)
#   datasets 4.7.0 -> 5.2.0       (upstream floor; lifts the fsspec cap)
#   torch_remat 0.2.2 -> git 6c05e8d8  (no release reports SavedTensorInfo.region)
#   mosaicml-streaming: PyPI 0.13.0 -> source build of main (0.14.0.dev0)
# Unchanged on purpose: grain 0.2.18, torch_checkpointing 0.1.1, pyarrow<26,
# torchdata 0.11.0, torchao 0.18.0, torchmonarch 0.7.0, tyro 1.0.17.
#
# =====================================================================
# Self-contained: all assets are inlined or downloaded from public upstreams.
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04@sha256:6435dc5a825b0095648d87a3c91240fd7788a85fafaf215739544d389ab74366

LABEL org.opencontainers.image.title="titaninfra-env" \
      org.opencontainers.image.description="titaninfra forward GPU environment (Ubuntu 24.04 + Python 3.12 + CUDA 13.2.1 + torch 2.15.0 + vLLM v0.33.0 built from source). Source-researched 2026-11-06; runtime verification is required before promotion. Project sources are mounted at runtime." \
      org.opencontainers.image.source="docker/titaninfra-full.Dockerfile" \
      org.titaninfra.build.stable_tag="TORCH_VERSION=2.15.0 TORCHVISION_VERSION=0.29.2 TORCHAUDIO_VERSION=2.11.1 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 VLLM_VERSION=v0.33.0" \
      org.titaninfra.build.tip_tag="Historical nightly metadata only; this standalone build uses stable Torch 2.15 and inline constraints."

# ---------------------------------------------------------------------
# Inherited header of InfraEnv.dockerfile (2026-09-05 upgrade record), kept
# verbatim as the version-rationale appendix. NOTE: its build-command block
# names the old file/tag and is superseded by the commands in the header above.
# ---------------------------------------------------------------------
# Current base: CUDA 13.2.1 + cuDNN + Ubuntu 24.04; the notes below are historical.
#
#     peft 0.20.0 / torchmonarch 0.6.0 / datasets 5.0.1 / helion 1.4.0 /
#     fla-core 0.5.2 / torchdata 0.11.0 / mosaicml-streaming 0.13.0 /
#     （flashinfer 0.6.16.post3 / tilelang 0.1.12 / apache-tvm-ffi 0.1.11 /
#     numba 0.65.0 / torchcodec>=0.14 / PyNvVideoCodec 2.0.4 / cutlass-dsl 4.6.2 /
#     quack-kernels 0.6.4 / tokenspeed-mla 0.1.8 / humming-kernels[cu13] 0.1.12）——
#

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# =====================================================================
# =====================================================================

# ---- Base system dependencies + Tsinghua mirror (ccache speeds up rebuilds) ----
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's@//.*security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get update && \
    apt-get install -y \
    software-properties-common \
    build-essential \
    cmake \
    ccache \
    git \
    wget \
    curl \
    vim \
    libibverbs-dev \
    libnuma-dev \
    librdmacm-dev \
    numactl \
    iproute2 \
    net-tools \
    pciutils \
    ethtool \
    gnupg \
    pdsh \
    python3-dev \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

ENV CCACHE_DIR=/root/.cache/ccache

# ---- Nsight Systems ----
RUN NSIGHT_VERSION=2025.6.1_2025.6.1.190-1_$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) && \
    wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_6/nsight-systems-${NSIGHT_VERSION}.deb && \
    apt-get update && apt-get install -y libxcb-cursor0 && \
    apt-get install -y ./nsight-systems-${NSIGHT_VERSION}.deb && \
    rm -rf /usr/local/cuda/bin/nsys && \
    ln -s /opt/nvidia/nsight-systems/2025.6.1/nsys /usr/local/cuda/bin/nsys && \
    rm -rf /usr/local/cuda/bin/nsys-ui && \
    ln -s /opt/nvidia/nsight-systems/2025.6.1/nsys-ui /usr/local/cuda/bin/nsys-ui && \
    rm nsight-systems-${NSIGHT_VERSION}.deb && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# MinIO client is isolated near the end so its network download cannot
# invalidate or block the compiled core; see INSTALL_MINIO_CLIENT below.

# ---- gdrcopy (CUDA-only library in the stable section; also used by DeepEP hybrid) ----
RUN git clone -b v2.5.2 https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy && \
    cd /tmp/gdrcopy && \
    make prefix=/usr/local lib_install && \
    cd / && rm -rf /tmp/gdrcopy

# ---- pip and the Python 3.12 symlink ----
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py --break-system-packages && \
    rm get-pip.py && \
    rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# ---- ssh ----
RUN apt-get update --fix-missing && \
    apt-get install -y ssh openssh-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# ---- uv: the version reported by the deployed container ----
ARG UV_VERSION=0.12.5
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uv==${UV_VERSION} && uv --version
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# ---- Mellanox OFED (user-only) ----
RUN cd /etc/apt/sources.list.d && \
    wget -q https://linux.mellanox.com/public/repo/mlnx_ofed/latest/ubuntu24.04/mellanox_mlnx_ofed.list && \
    sed -i 's|deb |deb [trusted=yes] |' mellanox_mlnx_ofed.list && \
    apt-get update && \
    MLNX_USER_PACKAGES="$(apt-cache show mlnx-ofed-all-user-only | \
        sed -n 's/^Depends: //p' | tr ',' '\n' | \
        sed 's/([^)]*)//g; s/^[[:space:]]*//; s/[[:space:]]*$//' | \
        sed '/^kernel-mft-dkms$/d')" && \
    test -n "$MLNX_USER_PACKAGES" && \
    case "$MLNX_USER_PACKAGES" in *kernel-mft-dkms*) echo 'DKMS exclusion failed' >&2; exit 1 ;; esac && \
    printf 'MLNX user-space packages (DKMS excluded): %s\n' "$MLNX_USER_PACKAGES" && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y $MLNX_USER_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# ---- JupyterLab and configuration ----
ENV PYTHON_VERSION=py3.12 \
    BASE_ENV_NAME=py3.12 \
    JUPYTER_ENV_NAME=py3.12
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    ipython \
    'jupyterlab>=4.1.0,<5.0.0a0' \
    jupyter \
    nbclassic \
    notebook \
    jupyterlab-lsp \
    anywidget \
    kaleido \
    pyviz_comms \
    lckr_jupyterlab_variableinspector \
    jupyterlab-spreadsheet-editor \
    jupyterlab-spreadsheet \
    jupyterlabcodetoc
RUN mkdir -p /root/.jupyter && \
    printf "c.ServerApp.disable_check_xsrf = True\nc.NotebookApp.disable_check_xsrf = True\nc.NotebookApp.token = ''\nc.ServerApp.token = ''\n" > /root/.jupyter/jupyter_notebook_config.py

# ---- Pure Python development and utility dependencies (stable section) ----
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    'expecttest>=0.2.0' pytest==9.1.1 pytest-cov pytest-asyncio pytest-xdist==3.8.0 pre-commit \
    pyrefly==0.45.1 ruff==0.16.3 rich tqdm av \
    pybind11 nvidia-mathdx \
    codetiming mathruler pylatexenc cachetools \
    nvtx matplotlib tensorboard wandb lancedb pylance \
    hydra-core dill pyarrow fsspec tyro einops pillow requests

# =====================================================================
# Limit local builds to the deployed H20 and the SM100 Dist-MoE target. Override
# with "8.0;8.6;8.9;9.0;10.0" for the former broader compatibility profile.
ARG TORCH_CUDA_ARCH_LIST="9.0;10.0"
# Conservative defaults for small local builders and x86 emulation. Override on
# larger build hosts; every compiled component, including FlashMLA, honors these.
ARG MAX_JOBS=1
ARG NVCC_THREADS=1
ARG CMAKE_BUILD_PARALLEL_LEVEL=1
ARG CCACHE_MAXSIZE=5G

#
#  Current stable constraints are embedded below and follow the Torch 2.15 wheel.
#  Historical nightly constraint files are neither required nor supported here.
# =====================================================================
RUN cat > /opt/constraints.txt <<'titaninfra_CONSTRAINTS_EOF'
#
#
#   triton ~=3.9.0 / nccl 2.30.7 / nvshmem 3.7.2 / cudnn 9.26.0.51
numpy>=2.1,<2.2
triton~=3.9.0
transformers==5.20.2
tokenizers==0.23.3
typing-extensions>=4.13.0
cuda-toolkit==13.2.2
cuda-bindings>=13.0.3,<14
nvidia-nccl-cu13==2.30.7
nvidia-nvshmem-cu13==3.7.2
nvidia-cudnn-cu13==9.26.0.51
nvidia-cutlass-dsl==4.8.0
apache-tvm-ffi==0.1.12
tilelang==0.1.15
pyarrow<26
# spmd_types 0.3.x is the first line that ships the `spmd_types.rules` submodule.
# torchtitan tip imports it unconditionally at package import time
# (torchtitan/models/common/linear/linear.py: `spmd.rules.einsum`,
# models/common/aux_loss.py: `spmd.rules.ignore` / `spmd.rules.output`), so 0.2.5
# fails with "module 'spmd_types' has no attribute 'rules'" on the first import.
# Verified against the published wheels: 0.2.5 has no rules.py, 0.3.1 and 0.3.2 do.
# 0.3.2 is the newest release (upstream's floor is 0.3.1) and also carries
# collective correctness fixes (inplace all_reduce contiguity, is_leaf handling,
# reduce_scatter contiguous input) that a training stack wants. Measured working
# on the shared GPU node with torch 2.15.0+cu132.
spmd_types==0.3.2
titaninfra_CONSTRAINTS_EOF

ARG TORCH_VERSION=2.15.0
ARG TORCHVISION_VERSION=0.29.2
ARG TORCHAUDIO_VERSION=2.11.1
# torch 2.15.0+cu132 METADATA requires triton~=3.9.0. The 2.16 nightly also used 3.9.
ARG TRITON_EXPECTED_PREFIX=3.9
# All three stable cu132 wheels are on the same index as of 2026-11-06; the former
# separate torchaudio test-channel install is no longer needed.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
ARG TORCH_WHEEL_URL=https://download.pytorch.org/whl/cu132/torch-2.15.0%2Bcu132-cp312-cp312-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_SHA256=84936484ed0a2e55fc89f29c7e580be2eff4fb4c9c2c46c88849cf8062281f96
ARG TORCH_WHEEL=/tmp/torch-2.15.0+cu132-cp312-cp312-manylinux_2_28_x86_64.whl
RUN --mount=type=cache,target=/root/.cache/pip \
    curl -fL --retry 8 --retry-all-errors --retry-delay 2 --connect-timeout 30 \
    "${TORCH_WHEEL_URL}" -o ${TORCH_WHEEL} && \
    echo "${TORCH_WHEEL_SHA256}  ${TORCH_WHEEL}" | sha256sum -c - && \
    pip install -c /opt/constraints.txt 'nvidia-nccl-cu13==2.30.7' 'triton~=3.9.0' \
    'cuda-bindings==13.4.4' 'cuda-pathfinder>=1.4.2' 'networkx>=2.5.1' \
    'sympy>=1.13.3' 'numpy==2.1.3' 'mpmath<1.4,>=1.1.0' && \
    pip install -c /opt/constraints.txt \
    ${TORCH_WHEEL} torchvision==${TORCHVISION_VERSION}+cu132 \
    torchaudio==${TORCHAUDIO_VERSION}+cu132 torchcodec==0.17.0+cu132 \
    --index-url ${TORCH_INDEX_URL} && \
    pip install -c /opt/constraints.txt "nvidia-ml-py>=12.560.30" && \
    # Torch 2.15 removed cuRAND from its cuda-toolkit extras; retain it explicitly
    # for compiled extensions that used to receive it transitively from torch.
    pip install -c /opt/constraints.txt 'cuda-toolkit[curand]==13.2.2' && \
    rm -f ${TORCH_WHEEL}
RUN python3 -c "import torch, triton; assert torch.__version__.startswith('${TORCH_VERSION}+cu132'), torch.__version__; assert torch.version.cuda.startswith('13.2'), torch.version.cuda; assert triton.__version__.startswith('${TRITON_EXPECTED_PREFIX}'), triton.__version__; print('  torch/triton OK:', torch.__version__, triton.__version__)"

# =====================================================================
# =====================================================================
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} \
    CCACHE_MAXSIZE=${CCACHE_MAXSIZE}

# ---- FlashInfer 0.7.2.post1 (vLLM v0.33.0 requirements/cuda.txt) ----
# The python/cubin pair is downloaded inside this layer and verified by SHA256.
# The pip index remains enabled for nccl-extensions and other runtime dependencies;
# --no-index would hide those requirements. Both exact wheel paths are installed
# explicitly so pip cannot silently select a different cubin release.
# JIT compilation against torch 2.15/cu132 still requires GPU verification.
# The pair moves as one: flashinfer's import-time guard compares the two
# distributions, so a single version ARG is the invariant rather than a shortcut.
ARG FLASHINFER_VERSION=0.7.2.post1
ARG FLASHINFER_PYTHON_SHA256=bfdc18b9e1e4d8d0730fe4343dc0e284cd0c160086de80f99888bc4867922d49
ARG FLASHINFER_CUBIN_SHA256=1998c7006bcc06ee5dd3208a682af247382c6710d6b370a961381262c04c11f0
RUN --mount=type=cache,target=/root/.cache/pip \
    curl -fL --retry 8 --retry-all-errors --retry-delay 2 --connect-timeout 30 \
    "https://github.com/flashinfer-ai/flashinfer/releases/download/v${FLASHINFER_VERSION}/flashinfer_cubin-${FLASHINFER_VERSION}-py3-none-any.whl" \
    -o /tmp/flashinfer_cubin-${FLASHINFER_VERSION}-py3-none-any.whl && \
    echo "${FLASHINFER_CUBIN_SHA256}  /tmp/flashinfer_cubin-${FLASHINFER_VERSION}-py3-none-any.whl" | sha256sum -c - && \
    pip download --no-deps --dest /tmp/flashinfer-python "flashinfer-python==${FLASHINFER_VERSION}" && \
    echo "${FLASHINFER_PYTHON_SHA256}  /tmp/flashinfer-python/flashinfer_python-${FLASHINFER_VERSION}-py3-none-any.whl" | sha256sum -c - && \
    pip install -c /opt/constraints.txt \
    /tmp/flashinfer-python/flashinfer_python-${FLASHINFER_VERSION}-py3-none-any.whl \
    /tmp/flashinfer_cubin-${FLASHINFER_VERSION}-py3-none-any.whl && \
    rm -rf /tmp/flashinfer-python /tmp/flashinfer_cubin-${FLASHINFER_VERSION}-py3-none-any.whl

# ---- TileLang + apache-tvm-ffi (2026-11-06) ----
# tilelang 0.1.15 requires tvm-ffi>=0.1.11,<0.1.13, so 0.1.12 is the newest
# coherent pair. It also satisfies attn-gym 0.0.20, FlashInfer and Dist-MoE.
# This deliberately updates beyond vLLM 0.33's exact 0.1.12/0.1.11 pair.
# tokenspeed-mla 2.0.3 no longer conflicts on tvm-ffi, but remains omitted to
# avoid introducing its separate tokenspeed-triton fork before MLA validation.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/constraints.txt tilelang==0.1.15 apache-tvm-ffi==0.1.12

# ---- DeepGEMM 2.8.2 (latest main, 2026-10-20) ----
# Historical torch 2.14 / v2.1.1 build notes; current setup.py already uses C++20.
# The newest DeepGEMM vendors `deep_jit`, whose `exception.hpp` includes
# <elfutils/libdwfl.h>; the older tag did not, so this header is a build requirement
# introduced by the upgrade itself (found by the first build after the bump: the
# compilation stopped on exactly this include). Kept next to the step that needs it
# rather than in the base apt block, so its reason stays visible.
RUN apt-get update && apt-get install -y --no-install-recommends libdw-dev \
    && rm -rf /var/lib/apt/lists/*

# 2026-10-07: the tag `v2.1.1.post3` (2025-10-15) is no longer the newest state --
# DeepGEMM publishes releases as commits on `main` ("Public release 26/09/30"), so the
# pin is the SHA. Previous pin kept here for the record: v2.1.1.post3.
# DeepGEMM 2.8.2 (2026-10-20). Both setup.py and CMake already select C++20;
# Ubuntu 24.04's GCC 13 supplies the required C++20 <format> implementation.
ARG DEEPGEMM_REF=5081f8513af57e61696de4f4f04b0e707ffb503e
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    pip install -c /opt/constraints.txt 'setuptools>=78.1.1,<81.0.0' wheel && \
    git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /opt/deepgemm-src && \
    cd /opt/deepgemm-src && git checkout ${DEEPGEMM_REF} && \
    python setup.py bdist_wheel && \
    pip install -c /opt/constraints.txt dist/*.whl && \
    cd / && python3 -c "import importlib.metadata as m; print('  deep-gemm', m.version('deep-gemm'))" && \
    rm -rf /opt/deepgemm-src /tmp/*

# ---- DeepEP main (v2, installed in one stage) ----
ENV GDRCOPY_HOME=/usr/local \
    NVSHMEM_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem
# 2026-11-06: still the newest DeepEP compatible with torch's NCCL pin. V2.5
# (`def865146e95`, 2026-09-29) and current HEAD require NCCL >= 2.32.3, while torch
# 2.15.0 requires `nvidia-nccl-cu13==2.30.7` exactly. NCCL is the training stack's ABI,
# not a build-only dependency, so do not override it for a DeepEP bump. This revision
# must be rebuilt and retested against torch 2.15's NVSHMEM 3.7.2.
# Previous pins: 93eb6eb23812 (2026-09-30, V2.5), 01dc3aaac820 (2026-08-04).
ARG DEEPEP_REF=a56d6156febcd9976e55adc85b5155bfac9f28f8
RUN cat > /opt/deepep-nvshmem36-p2p.patch <<'DEEPEP_PATCH_EOF'
diff --git a/csrc/kernels/legacy/internode_ll.cu b/csrc/kernels/legacy/internode_ll.cu
--- a/csrc/kernels/legacy/internode_ll.cu
+++ b/csrc/kernels/legacy/internode_ll.cu
@@ -280,8 +280,21 @@ __global__ __launch_bounds__(1024, 1) void dispatch(void* packed_recv_x,
     } else if (warp_id == num_warps - 1) {
         EP_DEVICE_ASSERT(num_sms > 1);
         if (sm_id == 0) {
-            // The first SM is also responsible for checking QPs
-            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);
+            // RC QPs are only needed for peers that are not directly reachable
+            // through P2P. NVSHMEM 3.6 zeroes num_rc_per_pe when its optional
+            // IBGDA transport is unavailable; all-local jobs still work because
+            // every send/count path below already falls back to the P2P pointer.
+            bool needs_ibgda = false;
+            for (int dst_rank = 0; dst_rank < num_ranks; ++dst_rank) {
+                if (dst_rank != rank and
+                    nvshmemi_get_p2p_ptr(
+                        reinterpret_cast<uint64_t>(rdma_recv_count), rank, dst_rank) == 0) {
+                    needs_ibgda = true;
+                    break;
+                }
+            }
+            EP_DEVICE_ASSERT(not needs_ibgda or
+                             ibgda_get_state()->num_rc_per_pe >= num_local_experts);

             // The first SM is also responsible for cleaning the next buffer
             #pragma unroll
DEEPEP_PATCH_EOF
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    pip install -c /opt/constraints.txt 'nvidia-nccl-cu13==2.30.7' && \
    # `--recurse-submodules` + the post-checkout update: the newest DeepEP vendors
    # `deep_jit` (its own submodule, `third-party/deep_jit/include`) and includes
    # <deep_jit/utils/no_ref_ptr.hpp>; the previous pin did not, so a plain shallow
    # clone built then and fails now. Both repositories vendor the same subtree, which
    # is why the two fixes in this file look alike.
    # The newest DeepEP vendors `deep_jit` as a submodule whose recorded URL is an SSH
    # one (git@github.com:deepseek-ai/DeepJIT.git), and a build container has no key.
    # `-c url...insteadOf` is applied to the clone itself as well as the later update:
    # `--recurse-submodules` fetches during the clone, so a config set afterwards is
    # too late (measured: the first attempt failed exactly there).
    git -c url."https://github.com/".insteadOf="git@github.com:" \
        clone --depth 1 --recurse-submodules https://github.com/deepseek-ai/DeepEP.git /opt/deepep && \
    cd /opt/deepep && git fetch --depth 1 origin ${DEEPEP_REF} && git checkout FETCH_HEAD && \
    git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive && \
    # NVSHMEM >=3.6 may leave num_rc_per_pe=0 without IBGDA; only require RC QPs
    # for peers without a P2P path (ported from vLLM PR #62519).
    git apply --check /opt/deepep-nvshmem36-p2p.patch && \
    git apply /opt/deepep-nvshmem36-p2p.patch && \
    export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH" && \
    export PATH="${NVSHMEM_DIR}/bin:$PATH" && \
    cd ${NVSHMEM_DIR}/lib && \
    ln -sf libnvshmem_host.so.3 libnvshmem_host.so && \
    cd /opt/deepep && \
    export TORCH_CUDA_ARCH_LIST=9.0 && \
    export CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:$CPATH && \
    export LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:$LIBRARY_PATH && \
    python setup.py install && \
    cd / && rm -rf /opt/deepep && \
    rm -rf /tmp/*

# ---- FlashMLA (vLLM 0.33.0's fork revision) ----
# Use e2afa5fd from vLLM's cmake/external_projects/flashmla.cmake. The fork carries
# DeepSeek V4.1 kernels and targets the torch >=2.10 stable ABI without upstream's
# post-build register-spill hard gate. This is source evidence, not a build receipt:
# CUDA 13.2 compilation and kernel numerics still require GPU verification.
# Fetch only the pinned commits. A full recursive clone pulls all CUTLASS history
# and failed twice on this network; shallow pinned fetches avoid that failure window.
ARG FLASHMLA_REPOSITORY=https://github.com/vllm-project/FlashMLA.git
ARG FLASHMLA_REF=e2afa5fd8c220fbd05fe03fc24c9d5e64c24641b
# Set to 1 for H20-only/resource-constrained builds; Dist-MoE keeps its own SM100 target.
ARG FLASH_MLA_DISABLE_SM100=0
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    git init /opt/flash_mla-src && \
    cd /opt/flash_mla-src && \
    git remote add origin ${FLASHMLA_REPOSITORY} && \
    (git fetch --depth 1 origin ${FLASHMLA_REF} || git fetch --depth 1 origin ${FLASHMLA_REF}) && \
    git checkout --detach FETCH_HEAD && \
    (git submodule update --init --recursive --depth 1 || \
     git submodule update --init --recursive --depth 1) && \
    FLASH_MLA_DISABLE_SM100=${FLASH_MLA_DISABLE_SM100} \
    CPATH=/usr/local/cuda/targets/x86_64-linux/include/cccl:/usr/local/cuda/include:$CPATH \
    MAX_JOBS=${MAX_JOBS} pip install --no-build-isolation . && \
    rm -rf /opt/flash_mla-src /tmp/*

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH} \
    PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/bin:${PATH}
WORKDIR /root

# =====================================================================
#  PART 4: latest stable vLLM source build (default v0.33.0 on torch 2.15.0)
#  This layer compiles v0.33.0 against torch 2.15.0 and applies the checked
#  NCCL2/cuBLAS compatibility patch. The v0.28/v0.31 notes below are historical
#  provenance only; the header describes current behavior.
#
#
#
# =====================================================================
ARG VLLM_VERSION=v0.33.0
ARG VLLM_TORCH_PIN_FROM=2.14.0
# v0.33.0 declares torch 2.14.0; build its C++/CUDA extensions against torch 2.15.0.
# The grep assertion turns the CMake compatibility edit into a required, checked step.
ARG VLLM_TORCH_PIN_TO=2.15.0
# Override to 9.0 on memory-constrained H20 builders; Dist-MoE retains SM100 separately.
ARG VLLM_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG VLLM_USE_EXISTING_TORCH=tools/use_existing_torch.py
# Use the stable tag's official cu130 wheel only for the Rust frontend. A local
# path avoids shallow-clone nightly fallback and urlretrieve's non-resumable download.
# v0.33.0 also treats any location as all-precompiled; the checked edit below removes
# only that implicit promotion so the explicit 0/1 extension/Rust flags stay authoritative.
ARG VLLM_PRECOMPILED_WHEEL_URL=https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl
ARG VLLM_PRECOMPILED_WHEEL_SHA256=4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa
ARG VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl
ENV VLLM_USE_PRECOMPILED=0 \
    VLLM_USE_PRECOMPILED_RUST=1 \
    VLLM_TARGET_DEVICE=cuda
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/constraints.txt ninja setuptools-scm setuptools-rust jinja2
ARG VLLM_DISABLE_MACHETE=0
RUN cat > /opt/vllm-0.33.0-torch215.patch <<'VLLM_TORCH215_PATCH_EOF'
diff --git a/vllm/distributed/parallel_state.py b/vllm/distributed/parallel_state.py
index 8975182..31cca16 100644
--- a/vllm/distributed/parallel_state.py
+++ b/vllm/distributed/parallel_state.py
@@ -25,6 +25,7 @@ If you only need to use the distributed environment without model/pipeline

 import contextlib
 import gc
+import os
 import pickle
 import weakref
 from collections import deque, namedtuple
@@ -313,6 +314,74 @@ def _platform_device_type() -> str:
         return "cpu"


+_PP_DEVICE_BACKEND = "nccl-lazy"
+
+
+def _resolved_device_backend(
+    torch_distributed_backend: str | Backend,
+) -> str:
+    backend = str(torch_distributed_backend)
+    if backend == "nccl":
+        default_pg = torch.distributed.distributed_c10d._get_default_group()
+        return default_pg._get_backend(torch.device("cuda")).name()
+    return backend
+
+
+def _pp_device_backend(
+    group_name: str, torch_distributed_backend: str | Backend
+) -> str | None:
+    backend = str(torch_distributed_backend)
+    if (
+        group_name != "pp"
+        or _platform_device_type() != "cuda"
+        or not backend.startswith("nccl")
+    ):
+        return None
+    resolved_backend = _resolved_device_backend(torch_distributed_backend)
+    if resolved_backend in {"nccl2", _PP_DEVICE_BACKEND}:
+        return _PP_DEVICE_BACKEND
+    return backend
+
+
+def _use_lazy_device_group(group_name: str, use_device_communicator: bool) -> bool:
+    # The vLLM world group is separate from PyTorch's default process group and
+    # is used for device communication only by optional features.
+    if group_name == "world":
+        return True
+    # Device communicators route normal tensor collectives through PyNCCL or a
+    # custom backend; a direct PyTorch user will initialize the device group on
+    # demand.
+    return use_device_communicator
+
+
+def _lazy_group_options(
+    torch_distributed_backend: str | Backend,
+) -> Any | None:
+    if _platform_device_type() != "cuda":
+        return None
+    if _resolved_device_backend(torch_distributed_backend) != "nccl2":
+        return None
+    options = torch.distributed.ProcessGroupNCCL.Options()
+    options.lazy_init = True
+    return options
+
+
+def _default_group_options(
+    torch_distributed_backend: str | Backend,
+) -> Any | None:
+    backend = str(torch_distributed_backend)
+    nccl2_selected = backend in {"nccl2", _PP_DEVICE_BACKEND} or (
+        backend == "nccl"
+        and hasattr(torch.distributed, "ProcessGroupNCCL2")
+        and os.environ.get("TORCH_DIST_USE_NCCL2") != "0"
+    )
+    if _platform_device_type() != "cuda" or not nccl2_selected:
+        return None
+    options = torch.distributed.ProcessGroupNCCL.Options()
+    options.lazy_init = True
+    return options
+
+
 def _device_backend_str(torch_distributed_backend: str | Backend) -> str:
     """Normalize ``torch_distributed_backend`` to the ``"<device>:<backend>"``
     format required by ``split_group``'s ``backend`` argument.
@@ -484,9 +553,24 @@ class GroupCoordinator:

         self.cpu_timeout = get_cpu_distributed_timeout_or_none()

+        pp_device_backend = _pp_device_backend(group_name, torch_distributed_backend)
+        use_lazy_device_group = _use_lazy_device_group(
+            group_name, use_device_communicator
+        )
+        lazy_group_options = (
+            _lazy_group_options(torch_distributed_backend)
+            if use_lazy_device_group
+            else None
+        )
+
         # VLLM_DISTRIBUTED_USE_SPLIT_GROUP gates the new ``split_group``
         # codepath. Default (False) preserves the legacy ``new_group`` path.
-        if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
+        if (
+            envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP
+            and pp_device_backend is None
+            and lazy_group_options is None
+            and all(len(ranks) > 1 for ranks in group_ranks)
+        ):
             self_device_group, self_cpu_group = _create_subgroups_split_group(
                 group_ranks, group_name, torch_distributed_backend
             )
@@ -500,10 +584,16 @@ class GroupCoordinator:
             device_timeout = get_distributed_timeout_or_none()

             for ranks in group_ranks:
+                pg_options = (
+                    _lazy_group_options(torch_distributed_backend)
+                    if len(ranks) == 1
+                    else lazy_group_options
+                )
                 device_group = torch.distributed.new_group(
                     ranks,
-                    backend=torch_distributed_backend,
+                    backend=pp_device_backend or torch_distributed_backend,
                     timeout=device_timeout,
+                    pg_options=pg_options,
                 )
                 # a group with `gloo` backend, to allow direct coordination between
                 # processes through the CPU.
@@ -1980,6 +2070,7 @@ def init_distributed_environment(
                 world_size=world_size,
                 rank=rank,
                 timeout=timeout,
+                pg_options=_default_group_options(backend),
             )
         if enable_elastic_ep:
             tp_pp_cpu_group = torch.distributed.new_group(
diff --git a/vllm/v1/worker/gpu_worker.py b/vllm/v1/worker/gpu_worker.py
index 8724faf..12d2454 100644
--- a/vllm/v1/worker/gpu_worker.py
+++ b/vllm/v1/worker/gpu_worker.py
@@ -653,6 +653,17 @@ class Worker(WorkerBase):
         """
         maybe_apply_startup_plan(self)

+        if current_platform.is_cuda_alike():
+            # cuBLAS(Lt) asks CUDA for its handle memory directly. When torch's own
+            # cudaMalloc fails, torch frees its unused cache and retries, but a cuBLAS
+            # request never triggers that because torch doesn't see it. Profiling can
+            # grow torch's cache until the GPU is full, and the first cuBLAS call would
+            # then fail even though torch holds idle memory it could have given back.
+            # Create the handles now, while memory is still free.
+            torch.cuda.current_blas_handle()
+            x = torch.ones(16, 16, dtype=torch.float16, device="cuda")
+            torch.nn.functional.linear(x, x, x[0])
+
         if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
             # still need a profile run which compiles the model for
             # max_num_batched_tokens
VLLM_TORCH215_PATCH_EOF
RUN cat > /opt/vllm-0.33.0-disable-machete.patch <<'VLLM_DISABLE_MACHETE_PATCH_EOF'
diff --git a/CMakeLists.txt b/CMakeLists.txt
index 7694e6f8..8c10a6f5 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -551,6 +551,10 @@ if(VLLM_TARGET_DEVICE STREQUAL "cuda" AND NOT VLLM_TARGET_DEVICE STREQUAL "cpu")
     # The machete kernels only work on hopper and require CUDA 12.0 or later.
     # Only build Machete kernels if we are building for something compatible with sm90a
     cuda_archs_loose_intersection(MACHETE_ARCHS "9.0a" "${CUDA_ARCHS}")
+    if(DEFINED ENV{VLLM_DISABLE_MACHETE} AND "$ENV{VLLM_DISABLE_MACHETE}" STREQUAL "1")
+      set(MACHETE_ARCHS "")
+      message(STATUS "Machete disabled by VLLM_DISABLE_MACHETE=1")
+    endif()
     if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND MACHETE_ARCHS)
       #
       # For the Machete kernels we automatically generate sources for various
VLLM_DISABLE_MACHETE_PATCH_EOF
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    curl -fL --retry 8 --retry-all-errors --retry-delay 2 --connect-timeout 30 \
    "${VLLM_PRECOMPILED_WHEEL_URL}" -o "${VLLM_PRECOMPILED_WHEEL_LOCATION}" && \
    echo "${VLLM_PRECOMPILED_WHEEL_SHA256}  ${VLLM_PRECOMPILED_WHEEL_LOCATION}" | sha256sum -c - && \
    test -f "${VLLM_PRECOMPILED_WHEEL_LOCATION}" && \
    git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm /tmp/vllm && \
    cd /tmp/vllm && \
    sed -i "s/TORCH_SUPPORTED_VERSION_CUDA \"${VLLM_TORCH_PIN_FROM}\"/TORCH_SUPPORTED_VERSION_CUDA \"${VLLM_TORCH_PIN_TO}\"/" CMakeLists.txt && \
    grep -q "TORCH_SUPPORTED_VERSION_CUDA \"${VLLM_TORCH_PIN_TO}\"" CMakeLists.txt && \
    test -f "${VLLM_USE_EXISTING_TORCH}" && \
    python "${VLLM_USE_EXISTING_TORCH}" && \
    git apply --check /opt/vllm-0.33.0-torch215.patch && \
    git apply /opt/vllm-0.33.0-torch215.patch && \
    case "$VLLM_DISABLE_MACHETE" in \
      0) ;; \
      1) git apply --check /opt/vllm-0.33.0-disable-machete.patch && git apply /opt/vllm-0.33.0-disable-machete.patch ;; \
      *) echo "VLLM_DISABLE_MACHETE must be 0 or 1" >&2; exit 1 ;; \
    esac && \
    grep -q '^        or bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION"))$' vllm/envs.py && \
    sed -i '/or bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION"))/d' vllm/envs.py && \
    printf 'vLLM build flags: extensions=%s rust=%s\n' "$VLLM_USE_PRECOMPILED" "$VLLM_USE_PRECOMPILED_RUST" && \
    test "$VLLM_USE_PRECOMPILED" = 0 && test "$VLLM_USE_PRECOMPILED_RUST" = 1 && \
    python3 -c "import importlib.util as i; s=i.spec_from_file_location('envs','vllm/envs.py'); m=i.module_from_spec(s); s.loader.exec_module(m); assert not m.VLLM_USE_PRECOMPILED; assert m.VLLM_USE_PRECOMPILED_RUST; print('parsed vLLM flags:', m.VLLM_USE_PRECOMPILED, m.VLLM_USE_PRECOMPILED_RUST)" && \
    VLLM_MAIN_CUDA_VERSION=13.0 VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION} \
    TORCH_CUDA_ARCH_LIST=${VLLM_CUDA_ARCH_LIST} \
    VLLM_USE_PRECOMPILED=0 VLLM_USE_PRECOMPILED_RUST=1 \
    VLLM_DISABLE_MACHETE=${VLLM_DISABLE_MACHETE} \
    pip install -v -c /opt/constraints.txt --no-build-isolation --no-deps . && \
    python3 -c "import importlib.metadata as m; v=m.version('vllm'); print('built vLLM version:', v); assert 'precompiled' not in v, v" && \
    cd / && rm -rf /tmp/vllm "${VLLM_PRECOMPILED_WHEEL_LOCATION}"

# ---- vLLM runtime closure (v0.33.0 requirements/common.txt + cuda.txt) ----
# Explicitly installed because the source build uses --no-deps. torch/vision/audio/
# codec are in PART 2; FlashInfer/TileLang/TVM-FFI are in PART 3. The detailed
# 0.31.0 diff below is historical provenance, not a description of current pins.
#
#     `oss-harmony >= 0.0.11  # Required for gpt-oss; includes embedded vocabularies`。
#   * `nvidia-cutlass-dsl[cu13]==4.6.2` → `==4.7.1`、`tilelang==0.1.12` →
#   * `pillow` / `einops` / `ninja` / `typing_extensions` / `requests` / `pyyaml` /
# Harmony: one import name, two mutually exclusive owners (see the note above), so
# the profile picks which is installed. The import check below is profile-independent:
# both provide `openai_harmony`, and both vLLM tags carry vllm/parser/harmony.py and
# vllm/entrypoints/chat_utils.py.
ARG VLLM_HARMONY_SPEC="oss-harmony==0.0.12"
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get remove -y python3-jwt || true && \
    pip install --ignore-installed PyJWT && \
    pip install -c /opt/constraints.txt \
    regex cachetools psutil sentencepiece 'requests>=2.33.0' tqdm blake3 py-cpuinfo \
    'huggingface_hub>=1.31.0,<3.0' 'safetensors>=0.6.2' \
    'protobuf>=5.29.6,!=6.30.*,!=6.31.*,!=6.32.*,!=6.33.0.*,!=6.33.1.*,!=6.33.2.*,!=6.33.3.*,!=6.33.4.*' \
    'fastapi[standard]>=0.133.0,<0.137.0' 'starlette>=1.0.1' 'aiohttp>=3.14.4' \
    'openai>=2.25.0' 'pydantic>=2.12.0' 'prometheus_client>=0.18.0' \
    'prometheus-fastapi-instrumentator>=8.0.0' 'tiktoken>=0.6.0' \
    lm-format-enforcer==0.11.3 'llguidance>=1.7.0,<1.8.0' outlines_core==0.2.14 \
    lark==1.2.2 'xgrammar==0.2.8' 'filelock>=3.20.3' partial-json-parser \
    'jsonschema>=4.23.0' 'pyzmq>=25.0.0' msgspec 'mistral_common[image]>=1.11.6' \
    'opencv-python-headless>=4.13.0' pyyaml six 'setuptools>=78.1.1,<81.0.0' \
    'pillow>=13.0.0' 'certifi>=2026.10.19' 'multidict>=6.9.1' 'orjson>=3.11.6' \
    'instanttensor>=0.2.2' 'typing_extensions>=4.13.0' \
    numba==0.65.0 PyNvVideoCodec==2.0.4 \
    'nvidia-cudnn-frontend>=1.31.0' nvtx==0.2.15 'fastsafetensors>=0.3.3' \
    'nvidia-cutlass-dsl[cu13]==4.8.0' quack-kernels==0.6.5 \
    'humming-kernels[cu13]==0.1.19' \
    compressed-tensors==0.17.0 depyf==0.20.0 cloudpickle watchfiles \
    python-json-logger pybase64 cbor2 ijson setproctitle \
    "${VLLM_HARMONY_SPEC}" 'anthropic>=0.71.0' \
    'model-hosting-container-standards>=0.1.14,<1.0.0' 'mcp>=2.2.0,<3.0.0' \
    'opentelemetry-sdk>=1.27.0' 'opentelemetry-api>=1.27.0' \
    'opentelemetry-exporter-otlp>=1.27.0' 'opentelemetry-semantic-conventions-ai>=0.4.1' && \
    python3 -c "import vllm, openai_harmony; print('  vllm', vllm.__version__, '/ harmony', openai_harmony.__file__)" && \
    python3 -c "import vllm.parser.harmony, vllm.entrypoints.chat_utils; print('  vllm harmony consumers OK')"

# ---- vllm-router (official PyPI wheel, SHA256-verified) ----
# Download inside this layer; no host prefetch or bind mount is required.
ARG VLLM_ROUTER_VERSION=0.1.17
ARG VLLM_ROUTER_SHA256=85c96bea4d573a49b5a8ae4c72ee3dfe733aba26b6f1ff1256e6142f1f42694d
RUN --mount=type=cache,target=/root/.cache/pip \
    pip download --no-deps --dest /tmp/vllm-router --index-url https://pypi.org/simple \
    "vllm-router==${VLLM_ROUTER_VERSION}" && \
    echo "${VLLM_ROUTER_SHA256}  /tmp/vllm-router/vllm_router-${VLLM_ROUTER_VERSION}-cp38-abi3-manylinux_2_28_x86_64.whl" | sha256sum -c - && \
    pip install --index-url https://pypi.org/simple \
    /tmp/vllm-router/vllm_router-${VLLM_ROUTER_VERSION}-cp38-abi3-manylinux_2_28_x86_64.whl && \
    rm -rf /tmp/vllm-router

# =====================================================================
# =====================================================================

# ---- mosaicml streaming: source build of main (0.14.0.dev0) -------------------
# Installed from source on request, not from the 0.13.0 wheel: main carries fixes
# that have no release yet (main is at 0.14.0.dev0 while the newest tag is v0.13.0).
# Reproducibility rule of this file still holds: the branch tip is pinned to the
# commit that was actually read, never to a floating "@main". Re-point
# STREAMING_REF (and STREAMING_VERSION_EXPECTED) when bumping, and re-read the
# dependency list below: the caps below are copied from setup.py AT THAT COMMIT,
# not guessed.
#   STREAMING_REF=1381ca1213fc95b6573efb44e5ec9f85c765ce1c  (main HEAD, 2026-11-03)
# Build: pure Python (no C/C++ extension), so the only build requirements are the
# image's setuptools<81+wheel -- already present -- and `--no-build-isolation`
# keeps pip from resolving a second setuptools against the network.
# Runtime deps are still installed explicitly (--no-deps build), from setup.py at
# the pinned commit. Two caps moved vs 0.13.0 and are the reason a wheel-era list
# cannot be reused: Brotli >=1.2.1 (was >=1.0.9) and paramiko <6 (was <4).
# Upstream's own declarations at this commit, for the record: numpy>=1.21.5,<2.5.0
# (image holds 2.1.3) and transformers>=4.21.3,<6 -- the former "transformers<5,
# deliberately violated" override is GONE: 5.20.2 now satisfies streaming itself.
# Note that `import streaming` executes streaming/__init__.py, which imports
# streaming.text -> `from transformers import AutoTokenizer`; so the import probe
# for this package lives in PART 7 (after transformers is pinned in PART 6), not
# here.
ARG STREAMING_REF=1381ca1213fc95b6573efb44e5ec9f85c765ce1c
ARG STREAMING_VERSION_EXPECTED=0.14.0.dev0
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get remove -y python3-cryptography python3-jwt || true && \
    pip install --ignore-installed PyJWT && \
    # Same shallow-two-step shape as the DeepEP block: clone the default branch,
    # then fetch the pinned commit explicitly. A plain full clone would pull the
    # whole history for a build that only needs one tree.
    git clone --depth 1 https://github.com/mosaicml/streaming.git /opt/streaming-src && \
    cd /opt/streaming-src && \
    git fetch --depth 1 origin ${STREAMING_REF} && \
    git checkout --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${STREAMING_REF}" && \
    rm -rf .git && \
    pip install -c /opt/constraints.txt --no-build-isolation --no-deps . && \
    cd / && rm -rf /opt/streaming-src && \
    pip install -c /opt/constraints.txt \
    'boto3>=1.21.45,<2' 'Brotli>=1.2.1' 'google-cloud-storage>=2.9.0,<3.3.0' \
    'paramiko>=2.11.0,<6' 'python-snappy>=0.6.1,<1' 'xxhash>=3.0.0,<4' \
    'zstd>=1.5.2.5,<2' 'oci>=2.88,<3' 'azure-storage-blob>=12.0.0,<13' \
    'azure-storage-file-datalake>=12.11.0,<13' 'azure-identity>=1.13.0' \
    'catalogue>=2,<3' && \
    python3 -c "import importlib.metadata as m; v=m.version('mosaicml-streaming'); assert v=='${STREAMING_VERSION_EXPECTED}', f'mosaicml-streaming {v} != ${STREAMING_VERSION_EXPECTED}'; print('  mosaicml-streaming', v, 'built from', '${STREAMING_REF}'[:12])" && \
    rm -rf /tmp/* /var/lib/apt/lists/*

# ---- torch ecosystem (2026-11-06): TRL 1.17.1 / Accelerate 1.15.0 / PEFT
# 0.21.2 / Liger 0.8.5 / TensorDict 0.15.5 / TorchFT 0.3.1 / Tyro 1.0.17.
# TorchAO 0.18.0 and TorchData 0.11.0 remain the latest stable releases.
# spmd_types is 0.3.2, not the 0.2.5 the old "three-way source/image gate" held:
# that gate pinned the image to a torchtitan revision that predates the
# spmd_types.rules submodule. torchtitan tip imports `spmd.rules` at module import
# time (models/common/linear/linear.py, models/common/aux_loss.py), so 0.2.5 now
# aborts the first model import. See the constraint block in PART 2 for the
# wheel-level evidence. ----
# Monarch 0.7 must move together with torchstore's StoreJob-based revision below;
# installing it with the legacy torchstore pin would break initialize_spmd.
ARG MONARCH_SPEC="torchmonarch==0.7.0"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/constraints.txt --no-deps trl==1.17.1 accelerate==1.15.0 peft==0.21.2 "${MONARCH_SPEC}" qwen_vl_utils && \
    pip install -c /opt/constraints.txt \
    'importlib-metadata>=6' \
    liger_kernel==0.8.5 tensordict==0.15.5 torchao==0.18.0 torchft==0.3.1 \
    'torchdata==0.11.0' \
    'tokenizers>=0.23.2,<0.24' safetensors 'huggingface_hub>=1.31.0,<3.0' \
    'spmd_types==0.3.2' 'tyro==1.0.17'

# ---- spmd_types surface self-check -------------------------------------------
# Turns the failure this audit started from into a build failure: every symbol
# below is imported by torchtitan tip's mandatory code paths, so a "looks like
# the right version" image that cannot serve them must not be produced.
# `spmd.rules.einsum` is the one that actually broke (linear.py passes `linear_in`
# which only exists in the 0.3.x rules kernel); get_local_type / reinterpret_mesh /
# has_local_type come from attention.py, token_dispatcher.py and deepseek_v4
# attention, PerMeshAxisSpmdType from context_parallel.py + hf_sharding.py.
RUN python3 - <<'PY'
import importlib.metadata as md
import spmd_types as spmd
from spmd_types import rules
from spmd_types.checker import typecheck  # noqa: F401

version = md.version("spmd_types")
assert version == "0.3.2", f"spmd_types {version} != 0.3.2"
for name in ("einsum", "ignore", "output", "NdimWithSpmdType"):
    assert hasattr(rules, name), f"spmd_types.rules has no {name}: {sorted(dir(rules))[:20]}"
for name in (
    "get_local_type",
    "has_local_type",
    "reinterpret_mesh",
    "register_local_autograd_function",
    "SpmdType",
    "PerMeshAxisSpmdType",
):
    assert hasattr(spmd, name), f"spmd_types has no top-level {name}"
import inspect

params = inspect.signature(rules.einsum).parameters
assert "linear_in" in params, f"rules.einsum lost linear_in: {sorted(params)}"
print(f"  spmd_types {version}: rules submodule + {len(dir(spmd))} public names OK")
PY

# ---- torchao prototype kernel autotune-key signature fix for an upstream bug (2026-11-15) ----
# `@triton.autotune(configs=atomic_kernel_configs_2D, key=["num_elements"])`
# `ValueError: key contains names that are not kernel arguments: num_elements`。
# `torchtitan.config.transform.quantization` → `torchtitan.quantization.float8`
RUN TORCHAO_F8ROWWISE_PY=/usr/local/lib/python3.12/dist-packages/torchao/prototype/moe_training/kernels/float8_rowwise.py && \
    grep -q 'key=\["num_elements"\]' "${TORCHAO_F8ROWWISE_PY}" && \
    sed -i 's/key=\["num_elements"\]/key=[]/' "${TORCHAO_F8ROWWISE_PY}" && \
    grep -q '@triton.autotune(configs=atomic_kernel_configs_2D, key=\[\])' "${TORCHAO_F8ROWWISE_PY}" && \
    echo "✓ torchao float8_rowwise autotune key bug patched (num_elements -> [])"

# ---- vime RL dependencies (Ray 2.58.0 to 2.60.0, PyPI 2026-11-04) ----
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/constraints.txt 'ray[default]==2.60.0' ring_flash_attn blobfile zstandard

# ---- TorchTitan development extras (helion 1.5.2 / fla-core 0.6.0) ----
#   Current ATTN_GYM_SPEC pins the released 0.0.20; the dependency surface stays checked.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/constraints.txt helion==1.5.2 fla-core==0.6.0 \
    'grain==0.2.18'

# =====================================================================
# =====================================================================
# Latest stable inside vLLM 0.33's transformers>=5.16.1,<5.21.0 range.
ARG TRANSFORMERS_SPEC="transformers==5.20.2"
ARG TOKENIZERS_SPEC="tokenizers==0.23.3"
ARG NUMPY_SPEC="numpy==2.1.3"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/constraints.txt --upgrade "${TRANSFORMERS_SPEC}" "${TOKENIZERS_SPEC}" && \
    pip install -c /opt/constraints.txt "${NUMPY_SPEC}" && \
    python3 -c "import transformers; assert transformers.__version__ == '5.20.2', f'transformers {transformers.__version__} != 5.20.2!'" && \
    echo "✓ transformers 5.20.2 verified" && \
    (pip check || echo "Informational pip check: compare reports with the deliberate overrides in env_scripts/README.md.")

# =====================================================================
#
#    titaninfra_env_p2_dockerfile   -> 8.2 torch_remat
#    titaninfra_env_p3_dockerfile   -> 8.3 mypy
#    titaninfra_env_p4_dockerfile   -> 8.4 renderers / radon / grimp
#
#
#  p6 first replaced PyPI 0.0.8 with a git pin. The current ATTN_GYM_SPEC instead
#  selects released 0.0.20 once, with [linear, cudnn, viz], and preserves the API checks.
#
# =====================================================================
ARG TORCH_REMAT_SPEC="torch_remat @ git+https://github.com/meta-pytorch/remat.git@6c05e8d8d6895a60dcc02674af0774b28155b5af"
# StoreJob was introduced in monarch 0.7 and is required by this torchstore revision.
# Stable torchx supplies monarch.job.spmd's specs/Runner imports. It is installed
# without nightly/dev extras and after all other pip installs in PART 8.5.
ARG TORCHX_SPEC="torchx==0.7.0"
ARG TORCHSTORE_SPEC="torchstore @ git+https://github.com/meta-pytorch/torchstore.git@e500a6ff6835d0ebc941c02e9302d0e4968c9a0c"
ARG REASONING_GYM_SPEC="reasoning-gym==0.1.25"
ARG MYPY_SPEC="mypy==2.4.0"
# Pin to the torchtitan tip's own pin, not to the number inside the image. On
# request (2026-11-18) this is the "synchronized source/config change" the old
# comment deferred to: torchtitan components/data/text_datasets.py and
# multimodal/mm_datasets.py import `build_training_sample`, `create_renderer` and
# the per-model renderer configs; 0.1.11 happens to carry those names, but the
# model families added since (muse_spark/muse_glimmer/qwen3_5) are developed
# against 0.1.12, which is also where CustomRendererConfig lands.
# Caveat kept explicit: titaninfra's renderers oracle (tests/oracle/
# *_renderers_oracle.py) was baselined on 0.1.11, so a renderers-sensitive diff
# in that oracle is expected to be re-baselined once, not treated as a bug.
ARG RENDERERS_SPEC="renderers==0.1.12"
# attn-gym 0.0.20 is upstream's pin and the minimum that works: torchtitan tip's
# models/common/attn_gym.py imports `attn_gym.mla.paged_mla_decode`, and the
# `attn_gym/mla` package does not exist in 0.0.17 (verified against both
# published wheels: 0.0.17 has linear/sparse/masks/mods, 0.0.20 adds mla +
# distributed + warpdrive). [linear,cudnn] is upstream's extra set; [viz] is kept
# for the image's existing notebook consumers.
ARG ATTN_GYM_SPEC="attn-gym[linear,cudnn,viz]==0.0.20"
# Latest main is required by torchtitan's MXFP8 fused-inference signature; v0.1.0
# lacks fuse_swiglu and weight_block_size. Import is CPU-safe; execution is SM100+.
ARG DIST_MOE_SPEC="dist-moe @ git+https://github.com/meta-pytorch/dist_moe.git@285763b5a0e01059a62c96b0990754a60a6ed94a"

# ---- 8.0 [new] the rest of upstream's declared dependency set ------------------
# Read at torchtitan HEAD ea6334273, .ci/docker/requirements.txt:
#   grain == 0.2.18              -> installed in PART 5, unchanged
#   datasets >= 5.2.0            -> here (was 4.7.0)
#   pyarrow < 26                 -> global constraint, unchanged
#   torch_checkpointing == 0.1.1 -> here, unchanged
#   spmd_types == 0.3.1          -> image ships 0.3.2, see PART 2 (0.2.5 was the bug)
# Why datasets moves: upstream raised the floor because "datasets caps fsspec in
# every release; 5.2.0 is the first to allow fsspec 2026.9.0". 4.7.0 declares
# fsspec<=2026.2.0, which caps a fleet-wide fsspec, and its huggingface-hub bound
# (<2.0) predates the >=1.31 floor vLLM needs. 5.2.0 requires pyarrow>=24 (the
# constraint keeps it <26), fsspec<=2026.9.0, huggingface-hub>=1.31 -- all of them
# compatible with the rest of this file. sanity: torchtitan's only datasets API is
# `datasets` + `datasets.distributed.split_dataset_by_node` in
# components/data/sources.py, present in both versions; the bump is for the
# transitive caps, not for a missing symbol.
ARG TORCH_CHECKPOINTING_SPEC="torch_checkpointing==0.1.1"
ARG DATASETS_SPEC="datasets==5.2.0"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/constraints.txt "${TORCH_CHECKPOINTING_SPEC}" "${DATASETS_SPEC}"
RUN python3 - <<'PYEOF'
import importlib.metadata as md

import torch_checkpointing  # noqa: F401

datasets = md.version("datasets")
major = int(datasets.split(".")[0])
assert major == 5, f"datasets {datasets} is below upstream's >=5.2.0 floor"
assert (major, int(datasets.split(".")[1])) >= (5, 2), f"datasets {datasets} < 5.2"
print(f"  torch_checkpointing {md.version('torch_checkpointing')} / datasets {datasets} OK")
PYEOF

# ---- 8.1 [merged from post_sync_layer.dockerfile] torchstore + reasoning-gym ----
# 2026-11-06 paired upgrade: torchstore e500a6ff uses monarch.job.spmd.StoreJob;
# monarch 0.6 lacks it, while the old torchstore's host_mesh_from_store is absent
# from monarch 0.7. Install --no-deps, then lock the cross-package API in one build.
RUN pip install -q --no-cache-dir --no-deps \
        "${TORCHSTORE_SPEC}" \
    && pip install -q --no-cache-dir --no-deps \
        pygtrie portpicker pyre-extensions typing-inspect mypy-extensions
RUN pip install -q --no-cache-dir "${REASONING_GYM_SPEC}"

# ---- 8.2 [merged from titaninfra_env_p2_dockerfile] torch_remat ----
#     python3 -c "import torch_remat" || pip install --user -q "torch_remat @ git+…"
#
# once torch_remat publishes one with SavedTensorInfo.region ... which per-region
# recompute=)、recompute_needs_tensor、is_recomputing、current_saved_tensor_info
#
RUN python3 -m pip install --no-cache-dir \
      "${TORCH_REMAT_SPEC}"
RUN python3 -c "import torch_remat, sys; print('torch_remat OK:', torch_remat.__file__); sys.exit(0)"

# ---- 8.3 [merged from titaninfra_env_p3_dockerfile] mypy ----
#
#
# Current mypy pin: 2.4.0, read from the deployed image on 2026-10-14.
# The former 2.3.1 helper result belongs to the historical layer. Type-checker
# versions affect gate results, so this image uses the observed canonical version.
#
# Ruff 0.16.3 is installed in PART 1 and is consumed by the release preflight.
# It is an image dependency; both its CLI and module entry point must work.
#
RUN python3 -m pip install --no-cache-dir "${MYPY_SPEC}"
RUN mypy --version && python3 -m mypy --version

# ---- 8.4 [merged from titaninfra_env_p4_dockerfile] renderers / radon / grimp ----
#
# * `radon` / `grimp` —— image pins 6.0.1 / 3.17, matching the deployed versions.
#
# * `pylance` / `av` / `pytest` / `xdist` / `lance` / `pyarrow` / `datasets` ——
#
# renderers declares openai-harmony, which would replace the oss-harmony module
# selected for vLLM. Install only its non-Harmony closure, then renderers itself
# with --no-deps; the final sole-owner assertion remains mandatory.
RUN python3 -m pip install --no-cache-dir -c /opt/constraints.txt \
    "radon==6.0.1" "grimp==3.17" jinja2 numpy \
    'openai>=1.108.1' 'prime-pydantic-config>=0.3.0.dev83' tiktoken && \
    python3 -m pip install --no-cache-dir --no-deps -c /opt/constraints.txt "${RENDERERS_SPEC}"
RUN python3 - <<'PY'
import radon, grimp, renderers
print("radon     ", getattr(radon, "__version__", "?"))
print("grimp     ", getattr(grimp, "__version__", "?"))
print("renderers ", getattr(renderers, "__version__", "?"))
PY

# ---- 8.5 [merged from titaninfra_env_p5_dockerfile + titaninfra_env_p6_dockerfile]
#      attn-gym: one released 0.0.20 install with [linear,cudnn,viz] extras ----
# Current release pin replaces the historical git/PyPI 0.0.8 installs below.
# 2026-11-18: 0.0.17 -> 0.0.20. Reason, read from torchtitan HEAD:
# torchtitan/models/common/attn_gym.py imports `attn_gym.mla.paged_mla_decode`
# (upstream commit "use attn-gym mla decode kernel", #5712), plus the gdn/kda
# `kernel_options` API from #6063. Wheel content check: 0.0.17 ships
# linear/sparse/masks/mods only -- there is no `attn_gym/mla` package at all --
# while 0.0.20 ships mla + distributed + warpdrive. So 0.0.17 cannot serve the
# torchtitan tip, independent of the pin text.
# Consumer note: attn_gym is now imported by *torchtitan* core (models/common/
# attn_gym.py, all 17 core import sites), not by titaninfra -- the titaninfra
# kimi_k3/qwen3_5 sites quoted further down no longer exist (verified by grep:
# zero `attn_gym` references under titaninfra/). The symbol locks below are kept
# because they still hold in 0.0.20 and they are the ones this project wired; the
# torchtitan surface is asserted separately at the end of this block.
#   `26222eb6922f051de70a8e1b98516ec9affd5338`（main HEAD, 2026-09-13T01:31:39Z）。
#   ContextParallelRouting / StagedOp / context_parallel_chunk /
#   context_parallel_conv_history）、`linear/kda/context_parallel.py`、
#   （pyproject.toml `[project.optional-dependencies] viz = [matplotlib,
# * The former -p6 git pin was superseded by ATTN_GYM_SPEC=0.0.17, then 0.0.20.
RUN python3 -m pip install --no-cache-dir -c /opt/constraints.txt "${ATTN_GYM_SPEC}"

# Dist-MoE is optional at runtime but pinned for the forthcoming torchtitan sync.
# --no-deps prevents its git build from replacing torch/triton; the explicit closure
# intersects dist_moe, attn-gym[cudnn], FlashInfer and vLLM at cutlass-dsl 4.8.0.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /opt/constraints.txt --no-deps "${DIST_MOE_SPEC}" && \
    pip install -c /opt/constraints.txt \
    'nvidia-cutlass-dsl[cu13]==4.8.0' 'cuda-python>=13.1,<14' \
    'apache-tvm-ffi>=0.1.10,<0.2' packaging
RUN python3 - <<'PYDISTMOE'
import importlib.metadata as md
import inspect
import json

import dist_moe

expected = "285763b5a0e01059a62c96b0990754a60a6ed94a"
dist = md.distribution("dist-moe")
direct_url = json.loads(dist.read_text("direct_url.json"))
assert direct_url["vcs_info"]["commit_id"] == expected, direct_url
signature = inspect.signature(dist_moe.prepare_block_scaled_weight)
assert {"fuse_swiglu", "weight_block_size"} <= set(signature.parameters), signature
print("  dist-moe", dist.version, direct_url["vcs_info"]["commit_id"], "API OK (execution requires SM100+)")
PYDISTMOE

# Install stable torchx last so no later resolver can honor its legacy urllib3<1.27
# pin and downgrade the modern vLLM stack. StoreJob does not import urllib3; pip
# check intentionally reports this metadata exception.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -q --no-cache-dir --no-deps -c /opt/constraints.txt "${TORCHX_SPEC}" && \
    python3 -c "import importlib.metadata as m; from monarch.job.spmd import StoreJob; import torchstore.spmd as spmd; assert spmd.StoreJob is StoreJob and hasattr(StoreJob, 'from_store'); assert m.packages_distributions()['torchx']==['torchx']; print('  torchstore / monarch 0.7 / stable torchx StoreJob API OK')"

RUN python3 - <<'PY'
import importlib.metadata as md
import inspect

version = md.version("attn-gym")
print("attn-gym version:", version)
assert version == "0.0.20", (
    f"expected attn-gym 0.0.20 (torchtitan tip's pin), got {version}; 0.0.17 has no "
    "attn_gym/mla package, and a stale attn-gym here means the linear-attention "
    "families silently fall back to warn-skip instead of failing the build"
)

# (0) torchtitan tip consumption surface (added 2026-11-18; this is the REAL reason
#     for 0.0.17 -> 0.0.20, and the only part of the bump that fails hard if the pin
#     is stale). Mirrors torchtitan/models/common/attn_gym.py's import list, minus the
#     deepest backend-specific private symbols (gdn/kda `*.impl.cudnn`, `*.impl.fused`,
#     `short_conv.cute`): those are TYPE_CHECKING-only in torchtitan and pulling the
#     cuDNN/cutlass-DSL backends into a CPU-only build step would add a failure mode
#     that has nothing to do with dependency completeness.
#     `attn_gym.mla` exists only in 0.0.20 (no such directory in the 0.0.17 wheel).
from attn_gym.linear import (  # noqa: F401
    KernelOptions,
    paged_chunk_gdn,
    recurrent_gdn,
    recurrent_gdn_decode,
)
from attn_gym.linear._delta_rule.gate import _FusedGate, gate_transform  # noqa: F401
from attn_gym.linear.gdn.ops import _ChunkGDN  # noqa: F401
from attn_gym.linear.kda import paged_chunk_kda, recurrent_kda_decode  # noqa: F401
from attn_gym.linear.kda.masking import _MaskRows  # noqa: F401
from attn_gym.mla import paged_mla_decode  # noqa: F401  # 0.0.20-only; the bump's reason
from attn_gym.sparse import AuxRequest, gather_attn, lightning_indexer  # noqa: F401

from attn_gym.linear.context_parallel import (  # noqa: F401
    ContextParallelPlan,
    ContextParallelRouting,
    StagedOp,
    context_parallel_chunk,
    context_parallel_conv_history,
)
from attn_gym.linear.gdn.context_parallel import context_parallel_gdn  # noqa: F401
from attn_gym.linear.gdn.stages import (  # noqa: F401
    chunk_gdn_prepare,
    chunk_gdn_prepare_backward,
)
from attn_gym.linear.kda.context_parallel import context_parallel_kda  # noqa: F401

from attn_gym.linear import causal_conv1d, chunk_gdn, l2norm  # noqa: F401  # qwen3_5/model.py:17-24
from attn_gym.linear.kda import bound_gate, chunk_kda  # noqa: F401  # kimi_k3/kda.py:13
from attn_gym.linear.kda.fwd.triton.l2norm_fwd import l2norm as kda_l2norm  # noqa: F401  # kda.py:14
from attn_gym.linear.short_conv import causal_conv1d as short_conv_conv1d  # noqa: F401  # kda.py:15

for name, fn in (
    ("context_parallel_gdn", context_parallel_gdn),
    ("context_parallel_kda", context_parallel_kda),
    ("context_parallel_conv_history", context_parallel_conv_history),
):
    print(f"  {name}{inspect.signature(fn)}")

REQUIRED_KWARGS = {
    "chunk_gdn": {"cu_seqlens", "scale", "impl"},
    "causal_conv1d": {"activation", "cu_seqlens", "initial_state"},
    # titaninfra/models/qwen3_5/model.py:314-315
    "l2norm": {"cu_seqlens"},
}
for name, fn in (("chunk_gdn", chunk_gdn), ("causal_conv1d", causal_conv1d), ("l2norm", l2norm)):
    params = inspect.signature(fn).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        print(f"  [warn] {name} accepts **kwargs; cannot statically confirm {sorted(REQUIRED_KWARGS[name])} is still used")
        continue
    missing = REQUIRED_KWARGS[name] - set(params)
    assert not missing, (
        f"{name} no longer accepts kwargs used by this project {sorted(missing)} (current main signature: {sorted(params)})"
    )

for dist in (
    # [linear,cudnn]
    "apache-tvm-ffi",
    "nvidia-cutlass-dsl",
    "cuda-python",
    # [viz]
    "marimo",
    "tabulate",
    "matplotlib",
):
    print(f"  {dist}: {md.version(dist)}")

print("8.5 self-check passed: attn-gym 0.0.20 (release pin), CP entry points, CP/kernel kwarg "
      "guards, torchtitan tip surface (attn_gym.mla + attn_gym.sparse) and linear/cudnn/viz "
      "extras all present.")
PY

# ---- 8.6 Final assertions for critical pins ----
#     renderers 0.1.12 / attn-gym 0.0.20 / datasets 5.2.x / mosaicml-streaming 0.14.0.dev0.
RUN python3 -c "import transformers, tokenizers; assert transformers.__version__=='5.20.2', transformers.__version__; assert tokenizers.__version__=='0.23.3', tokenizers.__version__; print('  transformers 5.20.2 / tokenizers 0.23.3 OK')" && \
    python3 -c "import grain; print('  grain OK', grain.__version__)" && \
    python3 -c "import importlib.metadata as m; assert m.version('apache-tvm-ffi')=='0.1.12', m.version('apache-tvm-ffi'); assert m.version('tilelang')=='0.1.15', m.version('tilelang'); print('  tvm-ffi 0.1.12 / tilelang 0.1.15 OK')" && \
    python3 -c "import importlib.metadata as m; names={d.metadata['Name'] for d in m.distributions()}; owners=[n for n in ('oss-harmony','openai-harmony') if n in names]; assert owners==['oss-harmony'], owners; print('  oss-harmony is the sole openai_harmony owner')" && \
    python3 -c "import importlib.util; assert importlib.util.find_spec('tokenspeed_mla') is None, 'tokenspeed-mla should be removed'; print('  tokenspeed-mla removed OK')" && \
    python3 -c "import importlib.metadata as m; v=m.version('spmd_types'); assert v=='0.3.2', v; v=m.version('renderers'); assert v=='0.1.12', v; v=m.version('attn-gym'); assert v=='0.0.20', v; print('  torchtitan pins: spmd_types 0.3.2 / renderers 0.1.12 / attn-gym 0.0.20')" && \
    python3 -c "import importlib.metadata as m; v=m.version('datasets'); assert tuple(int(x) for x in v.split('.')[:2]) >= (5,2), v; s=m.version('mosaicml-streaming'); assert s=='0.14.0.dev0', s; print(f'  data stack: datasets {v} / mosaicml-streaming {s} (built from main)')" && \
    python3 -c "import importlib.metadata as m, json; d=json.loads(m.distribution('torch_remat').read_text('direct_url.json')); c=d['vcs_info']['commit_id']; assert c.startswith('6c05e8d8'), c; print('  torch_remat from git', c[:12])" && \
    python3 -c "import torchstore; print('  torchstore OK', torchstore.__file__)" && \
    python3 -c "import reasoning_gym; print('  reasoning_gym OK', reasoning_gym.__version__)" && \
    echo "merged increment layers (PART 8) verified."

# =====================================================================
# =====================================================================
# ---- FlashInfer runtime warmup (no official cu132 JIT cache; run once on first GPU start,
#      JIT-compiling and autotuning common kernels to avoid first-request latency) ----
# ---- Layered post-build verification (T1 versions/ABI, T2 CUDA libraries, T3 ecosystem, T4 real GPU forward) ----
RUN cat > /opt/warmup_flashinfer.py <<'WARMUP_FLASHINFER_EOF'
#!/usr/bin/env python3
"""Warm up FlashInfer kernels on the first GPU-enabled container start.

Run common kernels through JIT compilation and autotuning, storing the cache in
FLASHINFER_JIT_CACHE_DIR (default: /opt/flashinfer_cache). Mount a persistent
volume or commit the container after the first run to avoid recompilation.

Usage inside a GPU-enabled container:
    python3 /opt/warmup_flashinfer.py [--head-dims 64 128 256] [--cache-dir /opt/flashinfer_cache]
"""
import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head-dims", type=int, nargs="+", default=[64, 128, 256],
        help="Head dimensions to warm up; adjust for your model",
    )
    parser.add_argument(
        "--cache-dir", default="/opt/flashinfer_cache",
        help="Writable JIT cache directory; a persistent volume is recommended",
    )
    args = parser.parse_args()

    os.environ.setdefault("FLASHINFER_JIT_CACHE_DIR", args.cache_dir)
    os.makedirs(args.cache_dir, mode=0o777, exist_ok=True)

    import torch

    assert torch.cuda.is_available(), "FlashInfer warmup requires a GPU"
    import flashinfer

    dev = torch.cuda.current_device()
    print(f"flashinfer {flashinfer.__version__} warmup on cuda:{dev}")

    for head_dim in args.head_dims:
        for dtype in (torch.float16, torch.bfloat16):
            num_heads, seq_len = 32, 64
            q = torch.randn(1, num_heads, seq_len, head_dim, device=dev, dtype=dtype)
            k = torch.randn(1, num_heads, seq_len, head_dim, device=dev, dtype=dtype)
            v = torch.randn(1, num_heads, seq_len, head_dim, device=dev, dtype=dtype)
            _ = flashinfer.single_prefill_with_kv_cache(
                q, k, v, causal=True, pos_encoding_mode="NONE", backend="auto"
            )
            print(f"  warm {dtype} × head_dim={head_dim} OK")

    print(f"flashinfer warmup done, cache dir: {args.cache_dir}")


if __name__ == "__main__":
    main()
WARMUP_FLASHINFER_EOF
RUN cat > /opt/verify_image.sh <<'VERIFY_IMAGE_EOF'
#!/usr/bin/env bash
#   bash /opt/verify_image.sh [--gpu]
set -uo pipefail

HAS_GPU=$([ "${1:-}" = "--gpu" ] && echo 1 || echo 0)
FAILED=0

step() { echo; echo "== $1 =="; }

check() {
    if python3 -c "$2" >/dev/null 2>&1; then
        echo "  ✓ $1"
    else
        echo "  ✗ $1"; FAILED=1
    fi
}

# ---- Tier 1: version and ABI consistency at import time ----
step "T1 versions and ABI"
check "torch 2.15.0+cu132" "import torch; assert torch.__version__.startswith('2.15.0+cu132'), torch.__version__"
check "cuda 13.2" "import torch; assert torch.version.cuda.startswith('13.2'), torch.version.cuda"
check "triton 3.9" "import triton; assert triton.__version__.startswith('3.9'), triton.__version__"
check "nccl 2.30.7" "import torch; assert tuple(torch.cuda.nccl.version()) == (2,30,7), torch.cuda.nccl.version()"
check "cudnn 9.x" "import torch; assert str(torch.backends.cudnn.version()).startswith('9'), torch.backends.cudnn.version()"
check "torchvision 0.29.2" "import torchvision; assert torchvision.__version__.startswith('0.29.2'), torchvision.__version__"
check "torchaudio 2.11.1" "import torchaudio; assert torchaudio.__version__.startswith('2.11.1'), torchaudio.__version__"
check "torchcodec 0.17.0" "import torchcodec; assert torchcodec.__version__.startswith('0.17.0'), torchcodec.__version__"
check "transformers 5.20.2" "import transformers; assert transformers.__version__ == '5.20.2', transformers.__version__"
check "tokenizers 0.23.3" "import tokenizers; assert tokenizers.__version__ == '0.23.3', tokenizers.__version__"
check "attn-gym 0.0.20" "import importlib.metadata as m; assert m.version('attn-gym')=='0.0.20', m.version('attn-gym')"
check "renderers 0.1.12 (torchtitan pin)" "import importlib.metadata as m; assert m.version('renderers')=='0.1.12', m.version('renderers')"
check "spmd_types 0.3.2 + rules submodule" "import importlib.metadata as m, spmd_types; assert m.version('spmd_types')=='0.3.2', m.version('spmd_types'); assert hasattr(spmd_types, 'rules') and hasattr(spmd_types.rules, 'einsum'), 'spmd_types.rules missing'"
check "datasets 5.2.x (upstream floor)" "import importlib.metadata as m; v=m.version('datasets'); assert tuple(int(x) for x in v.split('.')[:2]) >= (5,2), v"
check "mosaicml-streaming 0.14.0.dev0 (source build)" "import importlib.metadata as m; assert m.version('mosaicml-streaming')=='0.14.0.dev0', m.version('mosaicml-streaming')"
check "torch_checkpointing 0.1.1" "import importlib.metadata as m; assert m.version('torch_checkpointing')=='0.1.1', m.version('torch_checkpointing')"
check "dist-moe import (execution requires SM100+)" "import dist_moe, inspect; assert {'fuse_swiglu','weight_block_size'} <= set(inspect.signature(dist_moe.prepare_block_scaled_weight).parameters)"
check "torchmonarch 0.7.0" "import importlib.metadata as m; assert m.version('torchmonarch')=='0.7.0', m.version('torchmonarch')"
check "torchstore StoreJob API (stable torchx)" "import importlib.metadata as m; from monarch.job.spmd import StoreJob; import torchstore.spmd as spmd; assert spmd.StoreJob is StoreJob and hasattr(StoreJob, 'from_store'); assert m.packages_distributions()['torchx']==['torchx']"
check "numpy 2.1.x" "import numpy; assert numpy.__version__.startswith('2.1'), numpy.__version__"

# ---- Tier 2: CUDA library imports and NVSHMEM paths ----
step "T2 CUDA library imports"
check "vllm" "import vllm; print(vllm.__version__)"
check "flashinfer" "import flashinfer"
# `import vllm` does NOT exercise flashinfer, and flashinfer's own import-time guard
# (`flashinfer/jit/env.py::_get_cubin_dir`) raises RuntimeError when the two
# distributions disagree -- measured 2026-10-08: python 0.7.0.post1 against a leftover
# cubin 0.6.16.post3 fails `import flashinfer` outright, while `import vllm` passes.
# Asserting equality turns "both wheels were pinned to the same version" into a gate
# instead of a convention; it is deliberately separate from the import above.
check "flashinfer python and cubin versions match" "import importlib.metadata as m; a=m.version('flashinfer-python'); b=m.version('flashinfer-cubin'); assert a==b, f'flashinfer-python {a} != flashinfer-cubin {b}'; print(a)"
# vLLM 0.33 declares oss-harmony>=0.0.11. It provides the openai_harmony import
# namespace, which must have exactly one distribution owner; never install the legacy
# openai-harmony wheel alongside it.
check "vLLM Harmony consumers (openai_harmony, parser.harmony, and chat_utils have one owner)" "import importlib.metadata as m, openai_harmony, vllm.parser.harmony, vllm.entrypoints.chat_utils; p=[n for n in ('oss-harmony','openai-harmony') if n in {d.metadata['Name'] for d in m.distributions()}]; assert len(p)==1, f'harmony owners installed: {p}'; assert p != ['oss-harmony'] or m.version('oss-harmony')=='0.0.12', m.version('oss-harmony'); print(p[0], m.version(p[0]))"
check "tilelang" "import tilelang"
check "deep_gemm" "import deep_gemm"
check "deep_ep" "import deep_ep"
check "flash_mla" "import flash_mla"
check "NVSHMEM path" "import nvidia.nvshmem; p=next(iter(nvidia.nvshmem.__path__)); assert p.endswith('nvidia/nvshmem'), p; print(p)"
# `vllm collect-env` needs a device (`Failed to infer device type` without --gpus), so it
# runs in the GPU pass below: measured 2026-10-07 -- it fails identically in the previous
# canonical image when run without a GPU, i.e. the failure is the environment, not the
# image. Kept as a real check, moved to where it can hold.

# ---- Tier 3: ecosystem packages ----
step "T3 ecosystem"
for mod in "liger_kernel" "torchft" "trl" "ray" "tensordict" "datasets" "helion" "fla" "streaming" "torchao" "spmd_types" "tyro" "av" "wandb" "xdist" "torchdata"; do
    check "import $mod" "import $mod"
done
(pip check || echo "Informational pip check: compare reports with the deliberate overrides in env_scripts/README.md.")

# ---- Tier 4: real GPU forward with Triton 3.9 and FlashInfer JIT ----
if [ "$HAS_GPU" = "1" ]; then
    step "T4 real GPU forward"
    check "vllm collect-env consistency" "import subprocess; subprocess.run(['vllm','collect-env'], check=True, capture_output=True)"
    # vLLM's engine core must be able to re-import `__main__`: with CUDA already
    # initialised by the two checks above, vLLM forces the `spawn` start method, and
    # spawn skips its `init_main_from_path` step only when the main module has no file
    # -- a heredoc's `<stdin>` has one that does not exist in the child, which killed
    # the engine core during startup. Feeding the same heredoc through `-c` gives the
    # engine core a main module it can start from, so the real multiprocess path runs.
    python3 -c "$(cat <<'PY'
import os
import torch
assert torch.cuda.is_available(), "GPU is unavailable"
import torch.nn as nn
m = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256)).cuda().half()
x = torch.randn(4, 256, device="cuda", dtype=torch.float16)
y = torch.compile(m)(x)
y.sum().backward()
print("  ✓ torch.compile forward/backward (triton 3.9 codegen)")
import flashinfer
q = torch.randn(1, 32, 64, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 32, 64, 128, device="cuda", dtype=torch.float16)
v = torch.randn(1, 32, 64, 128, device="cuda", dtype=torch.float16)
_ = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, pos_encoding_mode="NONE")
print("  ✓ flashinfer JIT prefill")
# Local weights only. Container egress cannot reach huggingface.co, so a hub repo id
# (the previous `facebook/opt-125m`) fails for a reason that says nothing about this
# image. The weights are a host asset: mount the host's `.docker_models` (at `/models`
# by default, read-only is fine) or point VERIFY_GPU_MODEL_DIR elsewhere. Qwen3.5-0.8B is
# the model the RL cases already load into vLLM from that mount. When the directory is
# absent the check skips with this reason (exit 3) instead of failing, and it never
# falls back to a network download.
from vllm import LLM, SamplingParams
model_dir = os.environ.get("VERIFY_GPU_MODEL_DIR", "/models/Qwen3.5-0.8B")
if not os.path.isdir(model_dir):
    print(f"  - vLLM generate skipped: no local model dir at {model_dir} "
          "(host asset, not an image guarantee)")
    raise SystemExit(3)
llm = LLM(model=model_dir, enforce_eager=False, max_model_len=256,
          gpu_memory_utilization=0.3)
out = llm.generate(["Hello, world"], SamplingParams(max_tokens=8))
print("  ✓ vLLM generate:", out[0].outputs[0].text)
PY
)"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "  ✓ All real GPU forward checks passed"
    elif [ $rc -eq 3 ]; then
        echo "  ✓ GPU real forward checks passed; vLLM generate skipped -- no local model dir (see above)"
    else
        echo "  ✗ Real GPU forward checks failed"; FAILED=1
    fi
else
    echo "  Skipped; rerun with --gpu to enable GPU checks"
fi

echo
if [ $FAILED -eq 0 ]; then
    echo "ALL CHECKS PASSED."
else
    echo "SOME CHECKS FAILED."
    exit 1
fi
VERIFY_IMAGE_EOF
RUN chmod +x /opt/warmup_flashinfer.py /opt/verify_image.sh

RUN python3 -c "import torch; print(f'  torch {torch.__version__}')" && \
    python3 -c "import torchvision; print(f'  torchvision {torchvision.__version__}')" && \
    python3 -c "import torchaudio; print(f'  torchaudio {torchaudio.__version__}')" && \
    python3 -c "import triton; print(f'  triton {triton.__version__}')" && \
    python3 -c "import torch; print(f'  cuda {torch.version.cuda} / cudnn {torch.backends.cudnn.version()} / nccl {torch.cuda.nccl.version()}')" && \
    python3 -c "import transformers; print(f'  transformers {transformers.__version__}')" && \
    python3 -c "import tokenizers; print(f'  tokenizers {tokenizers.__version__}')" && \
    python3 -c "import numpy; print(f'  numpy {numpy.__version__}')" && \
    python3 -c "import vllm; print(f'  vllm {vllm.__version__}')" && \
    python3 -c "import flashinfer; print('  flashinfer OK')" && \
    (python3 -c "import deep_gemm; print('  deep_gemm OK')" || echo "  deep_gemm: import deferred to GPU verification (may need GPU runtime)") && \
    (python3 -c "import deep_ep; print('  deep_ep', deep_ep.__version__)" || echo "  deep_ep: import deferred to GPU verification (may need GPU runtime)") && \
    python3 -c "import tilelang; print('  tilelang OK')" && \
    (python3 -c "import flash_mla; print('  flash_mla OK')" || echo "  flash_mla: import deferred to GPU verification (may need GPU runtime)") && \
    python3 -c "import streaming; import importlib.metadata as m; print('  mosaicml-streaming', m.version('mosaicml-streaming'))" && \
    python3 -c "import fastapi; print(f'  fastapi {fastapi.__version__}')" && \
    python3 -c "import uvicorn; print(f'  uvicorn {uvicorn.__version__}')" && \
    python3 -c "import PIL; print(f'  Pillow {PIL.__version__}')" && \
    python3 -c "import pydantic; print(f'  pydantic {pydantic.__version__}')" && \
    python3 -c "import prometheus_client; print('  prometheus_client OK')" && \
    python3 -c "import pynvml; print('  pynvml OK')" && \
    python3 -c "import sentencepiece; print('  sentencepiece OK')" && \
    python3 -c "import tiktoken; print('  tiktoken OK')" && \
    python3 -c "import accelerate; print(f'  accelerate {accelerate.__version__}')" && \
    python3 -c "import peft; print(f'  peft {peft.__version__}')" && \
    python3 -c "import trl; print(f'  trl {trl.__version__}')" && \
    python3 -c "import wandb; print(f'  wandb {wandb.__version__}')" && \
    python3 -c "import datasets; print(f'  datasets {datasets.__version__}')" && \
    python3 -c "import jupyterlab; print(f'  jupyterlab {jupyterlab.__version__}')" && \
    python3 -c "import helion; import importlib.metadata as m; print('  helion', m.version('helion'))" && \
    python3 -c "import fla; print('  fla (fla-core) OK')" && \
    python3 -c "import attn_gym; print('  attn_gym OK')" && \
    python3 -c "import dist_moe; print('  dist_moe import OK (execution requires SM100+)')" && \
    python3 -c "import grain; print('  grain', grain.__version__)" && \
    python3 -c "import xdist; print('  pytest-xdist OK')" && \
    python3 -c "import av; print(f'  av {av.__version__}')" && \
    python3 -c "import ray; print('  ray', ray.__version__)" && \
    python3 -c "import torchao; print('  torchao', torchao.__version__)" && \
    python3 -c "import liger_kernel; print('  liger_kernel OK')" && \
    python3 -c "import torchft; print('  torchft OK')" && \
    python3 -c "import spmd_types; print('  spmd_types OK')" && \
    ruff --version && \
    python3 -m ruff --version && \
    ccache -s | head -4 && \
    echo "All critical imports verified successfully."

# ---- Optional MinIO client (last network-only layer) ----
# dl.min.io returns 410 after the OSS archival. Default builds install the final
# official GitHub release; constrained builders may defer it with
# --build-arg INSTALL_MINIO_CLIENT=0 without invalidating the compiled core.
ARG INSTALL_MINIO_CLIENT=1
ARG MC_VERSION=RELEASE.2025-08-13T08-35-41Z
ARG MC_SHA256=01f866e9c5f9b87c2b09116fa5d7c06695b106242d829a8bb32990c00312e891
RUN case "$INSTALL_MINIO_CLIENT" in \
    1) curl -fsSL --retry 4 --retry-all-errors \
        "https://github.com/minio/mc/releases/download/${MC_VERSION}/mc.linux-amd64.${MC_VERSION}" \
        -o /usr/local/bin/mc && \
       echo "${MC_SHA256}  /usr/local/bin/mc" | sha256sum -c - && \
       chmod +x /usr/local/bin/mc ;; \
    0) echo "MinIO client deferred (INSTALL_MINIO_CLIENT=0)" ;; \
    *) echo "INSTALL_MINIO_CLIENT must be 0 or 1" >&2; exit 1 ;; \
    esac

# ---- Final cleanup ----
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /root/.cache /var/tmp/*
