#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — GCE VM environment bootstrap
# =============================================================================
# Run once after provisioning a new L4 or A100 GCE VM.
# Requires root (sudo) for the CUDA 13.x apt install.
#
# What it does:
#   1. Installs CUDA Toolkit 13.2 system-wide via apt
#   2. Creates / updates the attn_bench conda environment
#   3. Installs the correct prebuilt flash-attn wheel
#   4. Runs a smoke test
#
# Usage:
#   bash setup_env.sh
# =============================================================================

set -euo pipefail

ENV_NAME="attn_bench"

# ── 1. Install CUDA Toolkit 13.2 via apt ─────────────────────────────────────
if nvcc --version 2>/dev/null | grep -q "release 13\."; then
  echo "[setup] CUDA 13.x already installed — skipping."
else
  echo "[setup] Installing CUDA Toolkit 13.2..."

  # Wait for any background apt/dpkg process (e.g. unattended-upgrades on
  # first boot) to release its lock before touching the package database.
  echo "[setup] Waiting for apt lock..."
  while sudo fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock \
               /var/cache/apt/archives/lock >/dev/null 2>&1; do
    echo "[setup]   apt is locked by another process — retrying in 5s..."
    sleep 5
  done
  echo "[setup] Lock free."

  # Add the CUDA apt repo for Ubuntu 22.04 (the standard GCE deep learning image)
  wget -q -O /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update -q
  # cuda-tileiras-13-2 and cuda-compiler-13-2 provide tileiras + nvcc without
  # pulling in the full toolkit (~4 GB).
  sudo apt-get install -y cuda-tileiras-13-2 cuda-compiler-13-2
  rm /tmp/cuda-keyring.deb
fi

# Put CUDA 13 tools on PATH for the rest of this script
export PATH="/usr/local/cuda-13.2/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "[setup] nvcc: $(nvcc --version | grep release)"

# ── 2. Create / update conda env ─────────────────────────────────────────────
echo ""
echo "[setup] Setting up conda environment '${ENV_NAME}'..."
if conda env list | grep -q "^${ENV_NAME} "; then
  conda env update --name "${ENV_NAME}" -f environment.yml --prune --solver=classic
else
  conda env create -f environment.yml --solver=classic
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── 3. Install flash-attn prebuilt wheel ──────────────────────────────────────
echo ""
echo "[setup] Installing FlashAttention-2..."
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_SHORT=$(python -c "import torch; print('cu'+''.join(torch.version.cuda.split('.')[:2]))")
PY_SHORT="cp$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")"
CXX_ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")

WHEEL="flash_attn-2.8.3+${CUDA_SHORT}torch${TORCH_VER}cxx11abi${CXX_ABI}-${PY_SHORT}-${PY_SHORT}-linux_x86_64.whl"
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHEEL}"

echo "[setup]   Trying prebuilt wheel: ${WHEEL}"
if pip install "${WHEEL_URL}" 2>/dev/null; then
  echo "[setup]   Installed from prebuilt wheel."
else
  echo "[setup]   Prebuilt wheel not found — compiling from source (~5 min)..."
  MAX_JOBS="${MAX_JOBS:-$(nproc)}" pip install flash-attn --no-build-isolation
fi

# ── 4. Write a shell snippet to auto-activate on login ───────────────────────
ACTIVATE_SNIPPET='
# attn_bench environment
export PATH="/usr/local/cuda-13.2/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
conda activate attn_bench 2>/dev/null || true
'
RC_FILE="$HOME/.bashrc"
if ! grep -q "attn_bench environment" "$RC_FILE" 2>/dev/null; then
  echo "$ACTIVATE_SNIPPET" >> "$RC_FILE"
  echo "[setup] Added auto-activation snippet to ${RC_FILE}."
fi

# ── 5. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "[setup] Smoke test..."
python - <<'PYEOF'
import torch, sys
assert torch.cuda.is_available(), "CUDA not available to PyTorch"
print(f"  python     {sys.version.split()[0]}")
print(f"  torch      {torch.__version__}")
print(f"  CUDA       {torch.version.cuda}")
print(f"  device     {torch.cuda.get_device_name(0)}")
try:
    import cuda.tile as ct; print("  cuda.tile  OK")
except ImportError as e: print(f"  cuda.tile  MISSING ({e})")
try:
    from flash_attn import flash_attn_func; print("  flash_attn OK")
except ImportError as e: print(f"  flash_attn MISSING ({e})")
PYEOF

echo ""
echo "[setup] Done. Open a new shell or run: source ~/.bashrc"