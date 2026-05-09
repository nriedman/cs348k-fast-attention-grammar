#!/usr/bin/env bash
# =============================================================================
# submit_benchmark.sh — Submit a single benchmark.py run to Slurm
#
# Usage:
#   ./submit_benchmark.sh --kernel kernels/my_kernel.py --output results/my_kernel.json [...]
#
# All flags are forwarded verbatim to benchmark.py.
# Slurm config is overridable via environment variables.
# =============================================================================

set -euo pipefail

PARTITION="${SLURM_PARTITION:-gpu}"
CONSTRAINT="${SLURM_CONSTRAINT:-}"
GPUS="${SLURM_GPUS:-1}"
CPUS_PER_GPU="${SLURM_CPUS_PER_GPU:-8}"
MEM_GB="${SLURM_MEM_GB:-64}"
TIME="${SLURM_TIME:-00:30:00}"
JOB_NAME="${SLURM_JOB_NAME:-attn_bench}"
CONDA_ENV="${CONDA_ENV:-attn_bench}"
LOG_DIR="${LOG_DIR:-logs}"

BENCH_ARGS="$*"

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CONSTRAINT_LINE=""
if [[ -n "$CONSTRAINT" ]]; then
  CONSTRAINT_LINE="#SBATCH --constraint=${CONSTRAINT}"
fi

sbatch <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=${GPUS}
#SBATCH --cpus-per-gpu=${CPUS_PER_GPU}
#SBATCH --mem=${MEM_GB}G
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}_%j.err
${CONSTRAINT_LINE}

echo "Job \$SLURM_JOB_ID on \$SLURMD_NODENAME — \$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV}

cd "\$(dirname "\$0")"
python benchmark.py ${BENCH_ARGS}
SBATCH_EOF

echo "[submit] Logs → ${LOG_DIR}/"