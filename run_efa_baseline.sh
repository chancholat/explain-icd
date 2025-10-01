#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --qos=priority
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=thin.nguyen@deakin.edu.au

# --- Environment setup ---
module load Anaconda3
source activate
conda activate haichan
export PYTHONNOUSERSITE=1
export WANDB_API_KEY=c76d20783a5b6c0eb844caaf78d65aef0e27d699

# --- Fixed base config ---
MAX_BATCH=16
BATCH=16
GPU=0

# --- Experiment grid ---
EXPERIMENTS=(mdace_icd9_code/plm_icd mdace_icd9_code/plm_icd_supervised mdace_icd9_code/plm_icd_pgd mdace_icd9_code/plm_icd_igr)

# --- Input arguments ---
exp_idx=$1   # 0..3

if [ -z "$exp_idx" ]; then
    echo "❌ Usage: sbatch run_plm.sh <exp_idx: 0..3>"
    exit 1
fi

if (( exp_idx < 0 || exp_idx >= ${#EXPERIMENTS[@]} )); then
    echo "❌ Invalid exp_idx: $exp_idx (must be 0..$(( ${#EXPERIMENTS[@]}-1 )))"
    exit 1
fi

EXPERIMENT="${EXPERIMENTS[$exp_idx]}"

# --- Logging ---
log_dir="logs"
mkdir -p "$log_dir"
STAMP="$(date +%Y%m%d-%H%M%S)"
JID="${SLURM_JOB_ID:-local$$}"
TAG="${EXPERIMENT}_baseline"

# sanitize experiment for filenames (replace / with _)
safe_exp="${EXPERIMENT//\//_}"

log_file="${log_dir}/plm_${safe_exp}_job${JID}_${STAMP}.out"
err_file="${log_dir}/plm_${safe_exp}_job${JID}_${STAMP}.err"
exec > "$log_file" 2> "$err_file"

echo "🚀 Starting ${TAG}"
echo "HOST: $(hostname)"
echo "JOB : ${JID}"
echo "---------------------------------------------"
echo "Fixed: MAX_BATCH=${MAX_BATCH}, BATCH=${BATCH}, GPU=${GPU}"
echo "Experiment: ${EXPERIMENT}"
echo "Logs: out=$log_file, err=$err_file"
echo "---------------------------------------------"

# --- Command ---
CMD=( python -u train_plm.py
  "experiment=${EXPERIMENT}"
  "dataloader.max_batch_size=${MAX_BATCH}"
  "dataloader.batch_size=${BATCH}"
  "gpu=${GPU}"
)

echo "[CMD] ${CMD[*]}"
echo "---------------------------------------------"

# --- Run ---
"${CMD[@]}"
status=$?

if [[ $status -eq 0 ]]; then
  echo "✅ Done: ${TAG}"
else
  echo "❌ Failed: ${TAG} (exit code $status)"
fi
