#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --qos=priority
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=thin.nguyen@deakin.edu.au

# ---------------- Env ----------------
module load Anaconda3
source activate
conda activate haichan
export PYTHONNOUSERSITE=1
export WANDB_API_KEY=c76d20783a5b6c0eb844caaf78d65aef0e27d699  # consider a secret manager

# ---------------- Fixed base config ----------------
MAX_BATCH=16
BATCH=16
GPU=0

# ---------------- Grids ----------------
EXPERIMENTS=(mdace_icd9_code/plm_icd mdace_icd9_code/plm_icd_supervised mdace_icd9_code/plm_icd_pgd mdace_icd9_code/plm_icd_igr) # $1 -> 0..3

# ---------------- Parse CLI args ----------------

exp_idx=$1

# Basic bounds checks
if (( exp_idx < 0 || exp_idx >= ${#EXPERIMENTS[@]} )); then
  echo "ERROR: exp_idx out of range (0..$(( ${#EXPERIMENTS[@]}-1 )))"; exit 3
fi

EXPERIMENT="${EXPERIMENTS[$exp_idx]}"

echo "=== JOB ${SLURM_JOB_ID:-N/A} on $(hostname) ==="
echo "Fixed: EXPERIMENT=${EXPERIMENT}, BATCH=${BATCH}, GPU=${GPU}"
echo "-----------------------------------------------"

mkdir -p logs

JID="${SLURM_JOB_ID:-local$$}"
STAMP="$(date +%Y%m%d-%H%M%S)"
# ---------------- One run ----------------
TAG="${EXPERIMENT}_baseline"
LOG="logs/plm_${TAG}_job${JID}_${STAMP}.out"
ERR="logs/plm_${TAG}_job${JID}_${STAMP}.err"

echo ""
echo ">> RUN ${TAG}"
echo "   -> log: ${LOG}"
echo "   -> err: ${ERR}"

CMD=( python train_plm.py
  "experiment=${EXPERIMENT}"
  "dataloader.max_batch_size=${MAX_BATCH}"
  "dataloader.batch_size=${BATCH}"
  "gpu=${GPU}"
)

{
  echo "JOB=${JID}"
  echo "HOST=$(hostname)"
  echo "[CMD] ${CMD[*]}"
  echo "-------------------------------------------"
} >"$LOG"

"${CMD[@]}" >>"$LOG" 2>"$ERR"

if [[ $? -eq 0 ]]; then
  echo "✅ Done: ${TAG}" | tee -a "$LOG"
else
  echo "❌ Failed: ${TAG} (see $ERR)" | tee -a "$LOG"
fi

echo "=== Completed for EXPERIMENT=${EXPERIMENT} ==="
