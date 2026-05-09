#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — One-shot environment bootstrap
# =============================================================================
# Run this ONCE on the cluster login node (or in an interactive job) before
# submitting benchmark jobs.  It creates the conda env, installs cuTile, and
# installs the correct prebuilt FlashAttention-2 wheel for your GPU/CUDA/Python
# triple.
#
# Usage:
#   bash setup_env.sh
#
# Prerequisites:
#   - CUDA Toolkit 13.1+ accessible (module or system install)
#   - conda / mamba available
# =============================================================================

set -euo pipefail

ENV_NAME="attn_bench"
PYTHON_VERSION="3.11"

# ── 0. Verify CUDA Toolkit ────────────────────────────────────────────────────
echo "[setup] Checking CUDA Toolkit..."
if ! command -v nvcc &>/dev/null; then
  echo ""
  echo "  [!] nvcc not found. cuTile requires CUDA Toolkit 13.1+."
  echo "  On a module-based cluster, try:"
  echo "      module avail cuda"
  echo "      module load cuda/13.x"
  echo "  On a Debian/Ubuntu system:"
  echo "      sudo apt-get install cuda-toolkit-13-2"
  echo "  On RHEL/Rocky:"
  echo "      sudo dnf install cuda-toolkit-13-2"
  echo ""
  exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
echo "[setup] Found CUDA Toolkit $CUDA_VERSION"

CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
if [[ "$CUDA_MAJOR" -lt 13 ]]; then
  echo "[!] WARNING: cuTile requires CUDA 13.1+. Found $CUDA_VERSION."
  echo "    FlashAttention-2 will still work on CUDA 11.8/12.x."
  echo "    Continuing, but cuda-tile install may fail."
fi

# ── 1. Create / update conda env ─────────────────────────────────────────────
echo "[setup] Creating conda environment '$ENV_NAME'..."
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "[setup] Environment already exists — updating."
  conda env update --name "$ENV_NAME" -f environment.yml --prune
else
  conda env create -f environment.yml
fi

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 2. Install FlashAttention-2 prebuilt wheel ────────────────────────────────
# flash-attn builds are specific to (Python, PyTorch, CUDA, cxx11-abi) —
# we detect each at runtime so the right wheel is fetched automatically.
echo ""
echo "[setup] Installing FlashAttention-2..."

TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_SHORT=$(python -c "import torch; v=torch.version.cuda; print('cu'+''.join(v.split('.')[:2]))")
PY_SHORT="cp$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")"
CXX_ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")

echo "[setup]   Python  : $PY_SHORT"
echo "[setup]   PyTorch : $TORCH_VERSION"
echo "[setup]   CUDA    : $CUDA_SHORT"
echo "[setup]   CXX ABI : $CXX_ABI"

# Try prebuilt wheel from the official GitHub releases first (much faster)
WHEEL_NAME="flash_attn-2.8.3+${CUDA_SHORT}torch${TORCH_VERSION}cxx11abi${CXX_ABI}-${PY_SHORT}-${PY_SHORT}-linux_x86_64.whl"
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHEEL_NAME}"

echo "[setup] Attempting prebuilt wheel: $WHEEL_NAME"
if pip install "$WHEEL_URL" 2>/dev/null; then
  echo "[setup] FlashAttention-2 installed from prebuilt wheel."
else
  echo "[setup] Prebuilt wheel not found for this combination."
  echo "[setup] Falling back to source compilation (this takes ~5 min with ninja)..."
  MAX_JOBS="${MAX_JOBS:-$(nproc)}" pip install flash-attn --no-build-isolation
fi

# ── 3. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "[setup] Running smoke test..."
python - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA not available to PyTorch"
print(f"  torch     {torch.__version__}")
print(f"  CUDA      {torch.version.cuda}")
print(f"  device    {torch.cuda.get_device_name(0)}")

try:
    import cuda.tile as ct
    print(f"  cuda.tile OK")
except ImportError as e:
    print(f"  cuda.tile MISSING ({e})")

try:
    from flash_attn import flash_attn_func
    print(f"  flash_attn OK")
except ImportError as e:
    print(f"  flash_attn MISSING ({e})")
PYEOF

echo ""
echo "[setup] Done. Activate the environment with:"
echo "    conda activate $ENV_NAME"