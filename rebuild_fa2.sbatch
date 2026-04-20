#!/bin/bash
# ============================================================
# Rebuild vllm-pace FULL — 16 CPUs, 12h budget
# Prior 8-CPU build timed out at 5h (FA3 Hopper kernels too slow)
# ============================================================
#SBATCH --job-name=rebuild-vllm-pace
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --account=gts-rs275-paid
#SBATCH --qos=inferno
#SBATCH --output=/storage/project/r-rs275-0/hlin464/vllm-pace/build_logs/rebuild_%j.out
#SBATCH --error=/storage/project/r-rs275-0/hlin464/vllm-pace/build_logs/rebuild_%j.err

set -euo pipefail

PROJECT_DIR="/storage/project/r-rs275-0/hlin464"
REPO_DIR="$PROJECT_DIR/vllm-pace"
ENV_DIR="$PROJECT_DIR/.conda_envs/vllm-pace"
SCRATCH="/storage/home/hcoda1/1/hlin464/scratch"

export PIP_CACHE_DIR="$SCRATCH/.pip_cache"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.9;9.0"
export VLLM_FA_CMAKE_GPU_ARCHES="80-real;86-real;89-real;90-real"
export VLLM_FA3_CMAKE_GPU_ARCHES="90a-real"
export MAX_JOBS=16
export NVCC_THREADS=2

module load miniforge/24.3.0-0 2>/dev/null || module load anaconda3 2>/dev/null || true
module load cuda/12.9.1
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"

echo "============================================================"
echo "vLLM PACE rebuild v3 — 16 CPUs, 12h budget"
echo "Node: $(hostname)  CPUs: $(nproc)  RAM: $(free -g | awk '/Mem:/{print $2}')G"
echo "nvcc: $(nvcc --version | tail -1)"
echo "Python: $(python --version)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "VLLM_FA_CMAKE_GPU_ARCHES=$VLLM_FA_CMAKE_GPU_ARCHES"
echo "VLLM_FA3_CMAKE_GPU_ARCHES=$VLLM_FA3_CMAKE_GPU_ARCHES"
echo "MAX_JOBS=$MAX_JOBS"
echo "============================================================"

cd "$REPO_DIR"

echo "[1/3] Cleaning stale artifacts..."
find . -name '_vllm_fa*_C*.so' -delete 2>/dev/null || true
rm -rf build/ *.egg-info/ 2>/dev/null || true

echo "[2/3] Full rebuild via pip..."
echo "Started: $(date)"

pip install --no-cache-dir --no-build-isolation -e . --no-deps \
    > "$REPO_DIR/build_logs/rebuild_${SLURM_JOB_ID}.log" 2>&1
BUILD_RC=$?

echo "Finished: $(date) (exit $BUILD_RC)"
tail -30 "$REPO_DIR/build_logs/rebuild_${SLURM_JOB_ID}.log"

if [ $BUILD_RC -ne 0 ]; then
    echo "[ERROR] Build failed:"
    grep -iE 'error:|fatal:|CMake Error' "$REPO_DIR/build_logs/rebuild_${SLURM_JOB_ID}.log" | grep -v diag_suppress | tail -30
    exit 1
fi

echo "[3/3] Verifying arch coverage..."
for f in "$REPO_DIR/vllm/_C.abi3.so" \
         "$REPO_DIR/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so" \
         "$REPO_DIR/vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so"; do
    echo "-- $(basename $f) --"
    cuobjdump --list-elf $f | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' '
    echo
done

cd /tmp
python -c "
import vllm
from vllm.vllm_flash_attn import _vllm_fa2_C, _vllm_fa3_C
print(f'vllm {vllm.__version__}')
print(f'FA2 import: OK')
print(f'FA3 import: OK')
"
echo "============================================================"
echo "DONE"
echo "============================================================"
