#!/bin/bash
# ============================================================
# Build vLLM v0.19.0 (upstream) in new conda env "vllm-pace"
# CPU cross-compilation for ALL PACE SM archs (excl. Blackwell)
# IMPORTANT: PyTorch is GPU-capable (cu126), NOT CPU-only
# ============================================================
#SBATCH --job-name=build-vllm-pace
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --account=gts-rs275-paid
#SBATCH --qos=inferno
#SBATCH --output=/storage/project/r-rs275-0/hlin464/vllm-pace/build_logs/build_%j.out
#SBATCH --error=/storage/project/r-rs275-0/hlin464/vllm-pace/build_logs/build_%j.err

set -euo pipefail

# ── paths ──
PROJECT_DIR="/storage/project/r-rs275-0/hlin464"
REPO_DIR="$PROJECT_DIR/vllm-pace"
ENV_DIR="$PROJECT_DIR/.conda_envs/vllm-pace"
SCRATCH="/storage/home/hcoda1/1/hlin464/scratch"

export PIP_CACHE_DIR="$SCRATCH/.pip_cache"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# ── SM archs: V100(7.0) RTX6000(7.5) A100(8.0) L40s(8.9) H100/H200(9.0) ──
# Blackwell excluded per user request
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.9;9.0"
export MAX_JOBS=8

# ── modules ──
module load miniforge/24.3.0-0 2>/dev/null || module load anaconda3 2>/dev/null || true
module load cuda/12.9.1 2>/dev/null || true
eval "$(conda shell.bash hook)"

echo "============================================================"
echo "Build vLLM v0.19.0 → env vllm-pace"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "RAM:  $(free -g | awk '/Mem:/{print $2}')G"
echo "nvcc: $(nvcc --version 2>&1 | tail -1)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "MAX_JOBS=$MAX_JOBS"
echo "============================================================"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: Clone vLLM v0.19.0 if needed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[1/5] Cloning vLLM v0.19.0..."
    git clone --branch v0.19.0 --depth 1 https://github.com/vllm-project/vllm.git "$REPO_DIR" 2>&1 | tail -3
else
    echo "[1/5] Repo already exists at $REPO_DIR"
fi
mkdir -p "$REPO_DIR/build_logs"
cd "$REPO_DIR"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: Create conda env
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ ! -d "$ENV_DIR/bin" ]; then
    echo "[2/5] Creating conda env (Python 3.11)..."
    conda create -p "$ENV_DIR" python=3.11 -y 2>&1 | tail -3
else
    echo "[2/5] Conda env already exists"
fi
conda activate "$ENV_DIR"

echo "Python: $(python --version)"
echo "pip:    $(pip --version)"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: Install PyTorch GPU (cu126) — NOT CPU!
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "[3/5] Installing PyTorch 2.10.0+cu126 (GPU version)..."
pip install --no-cache-dir \
    torch==2.10.0+cu126 \
    torchvision==0.25.0+cu126 \
    torchaudio==2.10.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126 \
    2>&1 | tail -5

# ── Verify it's CUDA, not CPU ──
python -c "
import torch
v = torch.__version__
cuda = torch.version.cuda
print(f'torch {v}, CUDA {cuda}')
assert 'cu' in v or cuda is not None, 'ERROR: CPU-only torch installed!'
assert '2.10.0' in v, f'ERROR: wrong torch version {v}'
print('✓ GPU PyTorch verified')
"

# ── Build dependencies ──
pip install --no-cache-dir \
    "setuptools>=77.0.3,<81" "setuptools-scm>=8" "vcs-versioning>=1.1" \
    cmake ninja jinja2 "packaging>=24.2" wheel regex build \
    "protobuf>=5.29.6" \
    2>&1 | tail -3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4: Build vLLM from source
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "[4/5] Building vLLM v0.19.0 (5 SM archs)..."
echo "Started: $(date)"

cd "$REPO_DIR"
rm -rf build/ *.egg-info/ 2>/dev/null || true

pip install --no-cache-dir --no-build-isolation -e . \
    > "$REPO_DIR/build_logs/vllm_build_${SLURM_JOB_ID}.log" 2>&1
BUILD_RC=$?

echo "Finished: $(date) (exit $BUILD_RC)"
tail -30 "$REPO_DIR/build_logs/vllm_build_${SLURM_JOB_ID}.log"

if [ $BUILD_RC -ne 0 ]; then
    echo "[ERROR] Build failed:"
    grep -i 'error\|fatal\|failed' "$REPO_DIR/build_logs/vllm_build_${SLURM_JOB_ID}.log" | tail -30
    exit 1
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5: Install additional packages (match continuum-exp-cpu)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "[5/5] Installing additional packages..."

# Core ML / serving
pip install --no-cache-dir \
    accelerate \
    "transformers>=4.56.0,<5" \
    datasets \
    einops \
    safetensors \
    sentencepiece \
    tokenizers \
    huggingface_hub \
    scipy \
    ray \
    xformers \
    2>&1 | tail -5

# SGLang
pip install --no-cache-dir \
    "sglang[all]>=0.5.10" \
    2>&1 | tail -5

# Flash Attention 4
pip install --no-cache-dir \
    flash-attn-4 \
    2>&1 | tail -5

# LMCache (for continuum experiments)
pip install --no-cache-dir \
    lmcache \
    2>&1 | tail -5

# Inference extras
pip install --no-cache-dir \
    compressed-tensors \
    lm-format-enforcer \
    xgrammar \
    outlines \
    llguidance \
    gguf \
    mistral_common \
    tiktoken \
    numba \
    cupy-cuda12x \
    cuda-python \
    torchao \
    2>&1 | tail -5

# Eval / misc
pip install --no-cache-dir \
    lm_eval \
    openai \
    anthropic \
    2>&1 | tail -5

# ── Final verification ──
echo "============================================================"
echo "Final verification"
echo "============================================================"
cd /tmp
python -c "
import torch
print(f'torch:        {torch.__version__} (CUDA {torch.version.cuda})')
assert 'cu' in torch.__version__ or torch.version.cuda is not None

import vllm
print(f'vllm:         {vllm.__version__}')

import transformers
print(f'transformers: {transformers.__version__}')

try:
    import sglang
    print(f'sglang:       {sglang.__version__}')
except: print('sglang:       not available')

try:
    import flash_attn_4
    print(f'flash-attn-4: OK')
except: print('flash-attn-4: not available')

try:
    import lmcache
    print(f'lmcache:      OK')
except: print('lmcache:      not available')

import vllm._C
print(f'vllm._C:      OK')

print()
print('BUILD SUCCESS — env: $ENV_DIR')
"
echo "============================================================"
