# Environment Setup

This document explains how to get all dependencies installed before running benchmark jobs on a Slurm cluster.

---

## Dependency overview

| Package | Why needed | Notes |
|---|---|---|
| Python 3.11 | Runtime | flash-attn has best wheel coverage on 3.11 |
| PyTorch 2.5 | Tensor ops, CUDA events | Must match the CUDA toolkit minor version |
| `cuda-tile` | Your grammar cross-compiles to CuTile | Requires CUDA Toolkit **13.1+** on the host |
| `flash-attn` | SOTA baseline kernel | Wheel is specific to (Python, PyTorch, CUDA) triple |
| `ninja` | C++ build parallelism | Required for flash-attn source compilation |

---

## Step 1 — Load a CUDA 13.1+ module

cuTile's compiler (`tileiras`) requires CUDA Toolkit 13.1 or newer. On most clusters the toolkit is not in `PATH` by default; you must load a module first.

```bash
module avail cuda          # list available versions
module load cuda/13.2      # or whatever 13.x version is available
nvcc --version             # confirm it's visible
```

If CUDA 13.x is not available as a module, ask your sysadmin or install the toolkit directly into your home directory via the NVIDIA runfile installer. CUDA 12.x is sufficient if you are not using cuTile yet — flash-attn works on CUDA 11.8/12.x.

---

## Step 2 — Bootstrap the conda environment

Run this once from the login node or an interactive job:

```bash
cd /path/to/eval
bash setup_env.sh
```

The script:
1. Checks that `nvcc` is present and warns if the version is below 13.1
2. Creates (or updates) the `attn_bench` conda environment from `environment.yml`
3. Detects your exact (Python, PyTorch, CUDA, cxx11-ABI) combination and installs the matching prebuilt `flash-attn` wheel directly from the official GitHub releases — no compilation needed in the common case
4. Falls back to source compilation via `pip install flash-attn --no-build-isolation` if a prebuilt wheel is not available for your combination (this takes ~5 minutes with `ninja` on a 32-core node)
5. Runs a smoke test that prints the versions of each key package and confirms CUDA is visible to PyTorch

---

## Step 3 — Verify the environment interactively (optional)

Before submitting batch jobs it is worth running a quick interactive test on a GPU node:

```bash
# Request a short interactive GPU session
srun --partition=gpu --gpus=1 --time=00:10:00 --pty bash

module load cuda/13.2
conda activate attn_bench

# Confirm everything is importable and CUDA is reachable
python - <<'PY'
import torch, cuda.tile as ct
from flash_attn import flash_attn_func
print("torch   :", torch.__version__)
print("device  :", torch.cuda.get_device_name(0))
print("cuda.tile OK")
print("flash_attn OK")
PY

# Run the benchmark against the naive kernel to confirm the harness works
python benchmark.py \
    --kernel kernels/naive.py \
    --label  naive \
    --seq    2048 \
    --output /tmp/smoke_test.json
```

---

## Step 4 — Submit benchmark jobs

Once the environment is verified, jobs can be submitted normally:

```bash
CONDA_ENV=attn_bench SLURM_PARTITION=gpu \
  ./submit_benchmark.sh \
    --kernel kernels/my_grammar_kernel.py \
    --label  iter_042 \
    --output results/iter_042.json
```

The Slurm job script activates the conda environment before running the benchmark, so no manual activation is needed at submission time.

---

## Updating the environment

If you add a new dependency (e.g. `triton`, `cupy`):

1. Add it to `environment.yml` under `pip:`
2. Re-run `bash setup_env.sh` — the update path (`conda env update --prune`) will apply the diff without rebuilding from scratch

---

## Troubleshooting

**`nvcc not found` inside the Slurm job**
Add `module load cuda/13.2` to `submit_benchmark.sh` before the conda activation line. The job environment does not inherit your login shell's loaded modules.

**`flash-attn` wheel not found / compilation fails**
Check that `ninja` is installed in the environment (`ninja --version`). If RAM is limited on the compile node, set `MAX_JOBS=4` before running `setup_env.sh` to cap parallel compilation workers.

**`cuda.tile` imports but kernels fail to compile at runtime**
cuTile compiles kernels JIT when first called. It needs `tileiras` (installed automatically with `cuda-tile`) and `ptxas` (from the CUDA Toolkit). Confirm both are on `PATH`:
```bash
which tileiras
which ptxas
```
If `ptxas` is missing, re-load the CUDA module or install `cuda-compiler-13-2` via apt/dnf.

**PyTorch and flash-attn CUDA version mismatch**
The CUDA version PyTorch was compiled against (shown by `torch.version.cuda`) must match the `cu...` suffix of the installed flash-attn wheel. `setup_env.sh` detects this automatically, but if you manually install a different PyTorch version afterwards, re-run `setup_env.sh` to reinstall the matching flash-attn wheel.