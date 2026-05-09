# Environment Setup

Setup for GCE VMs — run `setup_env.sh` once after provisioning, then use `benchmark.py` directly.

---

## Dependency overview

| Package | Why needed | Notes |
|---|---|---|
| Python 3.11 | Runtime | Best flash-attn wheel coverage |
| PyTorch 2.5 | Tensor ops, CUDA events | Built against CUDA 12.1 |
| `cuda-tile` | Grammar cross-compiles to cuTile | Needs CUDA Toolkit 13.1+ system-wide |
| `flash-attn` | SOTA baseline kernel | Wheel is specific to (Python, PyTorch, CUDA) triple |
| `ninja` | C++ build parallelism | Fallback if no prebuilt flash-attn wheel exists |

---

## VM provisioning

Use the **NVIDIA GPU-Optimized Image** from the GCE Marketplace (search "NVIDIA GPU-Optimized VMI"). It ships with the NVIDIA driver, CUDA 12.x, and conda pre-installed, which means:

- The GPU driver is already present — `setup_env.sh` only adds the CUDA 13.x toolkit on top.
- No manual driver install is needed between the L4 and A100 VMs; the same image works for both.

Recommended machine types:
- **Development (L4):** `g2-standard-8` (1× L4, 8 vCPU, 32 GB RAM) — ~$0.70/hr
- **Evaluation (A100):** `a2-highgpu-1g` (1× A100 40GB, 12 vCPU, 85 GB RAM) — ~$3.60/hr

**Stop (don't delete) the A100 VM between eval runs** to avoid re-running setup. The L4 VM can stay running during development since the cost is low.

---

## Step 1 — Run setup_env.sh

SSH into the VM and run:

```bash
git clone <your-repo>
cd <your-repo>/eval
bash setup_env.sh
```

The script:
1. Installs `cuda-tileiras-13-2` and `cuda-compiler-13-2` via apt (provides `tileiras` and `nvcc` without downloading the full ~4 GB toolkit)
2. Creates the `attn_bench` conda environment from `environment.yml`
3. Detects your exact (Python, PyTorch, CUDA, cxx11-ABI) triple and installs the matching prebuilt `flash-attn` wheel; falls back to source compilation if no wheel is found (~5 min with `ninja`)
4. Appends a CUDA 13.2 `PATH` / `LD_LIBRARY_PATH` export and `conda activate attn_bench` to `~/.bashrc` so every new shell is ready to go
5. Runs a smoke test

This is the **same script on both VMs** — run it once per VM after provisioning.

---

## Step 2 — Run benchmarks

After setup, benchmarks are just direct Python invocations:

```bash
python benchmark.py \
    --kernel kernels/my_kernel.py \
    --label  iter_042 \
    --seq    2048 \
    --output results/iter_042.json
```

No Slurm, no job submission — you have the whole GPU to yourself.

---

## Updating the environment

If you add a dependency:

1. Add it to `environment.yml` under `pip:`
2. Re-run `bash setup_env.sh` — the update path (`conda env update --prune`) applies the diff in-place

---

## Troubleshooting

**`cuda.tile` imports but kernels fail to compile at runtime**
cuTile compiles JIT on first call and needs both `tileiras` and `ptxas` on `PATH`. Confirm:
```bash
which tileiras && which ptxas
```
If either is missing, the CUDA 13.2 apt packages may not have installed cleanly. Re-run:
```bash
sudo apt-get install -y cuda-tileiras-13-2 cuda-compiler-13-2
```

**`flash-attn` wheel not found / compilation fails**
Confirm `ninja` is available (`ninja --version`). If the VM has limited RAM, cap parallel jobs:
```bash
MAX_JOBS=4 bash setup_env.sh
```

**PyTorch and flash-attn CUDA version mismatch**
The CUDA version PyTorch was compiled against (`torch.version.cuda`) must match the `cu...` suffix in the installed flash-attn wheel. `setup_env.sh` detects this automatically. If you manually change the PyTorch version afterwards, re-run `setup_env.sh` to reinstall the matching wheel.

**`nvidia-smi` shows the driver but `torch.cuda.is_available()` returns False**
The NVIDIA driver is present but the CUDA runtime isn't on `LD_LIBRARY_PATH`. Open a new shell (so `~/.bashrc` is sourced) or run:
```bash
source ~/.bashrc
```