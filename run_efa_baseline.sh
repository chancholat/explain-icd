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

# --- Experiment list ---
EXPERIMENTS=(
  "mdace_icd9_code/plm_icd"
  "mdace_icd9_code/plm_icd_supervised"
  "mdace_icd9_code/plm_icd_pgd"
  "mdace_icd9_code/plm_icd_igr"
)

# --- Logging ---
log_dir="logs"
mkdir -p "$log_dir"
STAMP="$(date +%Y%m%d-%H%M%S)"
JID="${SLURM_JOB_ID:-local$$}"

# --- Loop through experiments ---
for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    TAG="${EXPERIMENT}_baseline"
    safe_exp="${EXPERIMENT//\//_}"   # replace / with _ for safe filenames

    log_file="${log_dir}/plm_${safe_exp}_job${JID}_${STAMP}.out"
    err_file="${log_dir}/plm_${safe_exp}_job${JID}_${STAMP}.err"
    exec > >(tee -a "$log_file") 2> >(tee -a "$err_file" >&2)

    echo "🚀 Starting experiment: ${EXPERIMENT}"
    echo "JOB: $JID on host $(hostname)"
    echo "MAX_BATCH=${MAX_BATCH}, BATCH=${BATCH}, GPU=${GPU}"
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
    echo "============================================="
done

echo "🎉 All experiments completed."
