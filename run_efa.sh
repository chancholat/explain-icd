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
EXPERIMENT="mdace_icd9_code/plm_efa"
MAX_BATCH=8
BATCH=8
GPU=0

# ---------------- Grids ----------------
# Injected via CLI (two indices only)
LRS=(5e-5 1e-5)                          # $1 -> 0..1
LAMBDAS=(0.0001 0.00001 0.000001 0.001)


# SOFT_ALPHAS=(0.5 0.3 0.1 0.01)           # $2 -> 0..3
SOFT_ALPHAS=(0.1)                # fixed for now
# METHODS=(laat grad_attention)
METHODS=(laat)              # fixed for now
REFS=("models/supervised/ym0o7co8" "models/supervised_attention_full_target")

# ---------------- Parse CLI args ----------------
if [[ $# -lt 2 ]]; then
  # echo "Usage: sbatch $0 <lr_idx:0-1> <soft_alpha_idx:0-3>"
  echo "Usage: sbatch $0 <lr_idx:0-1> <lambda_idx:0-3>"
  exit 1
fi

lr_idx="$1"
# sa_idx="$2"
lambda_idx="$2"

# Basic bounds checks
if (( lr_idx < 0 || lr_idx >= ${#LRS[@]} )); then
  echo "ERROR: lr_idx out of range (0..$(( ${#LRS[@]}-1 )))"; exit 2
fi
# if (( sa_idx < 0 || sa_idx >= ${#SOFT_ALPHAS[@]} )); then
#   echo "ERROR: soft_alpha_idx out of range (0..$(( ${#SOFT_ALPHAS[@]}-1 )))"; exit 3
# fi
if (( lambda_idx < 0 || lambda_idx >= ${#LAMBDAS[@]} )); then
  echo "ERROR: lambda_idx out of range (0..$(( ${#LAMBDAS[@]}-1 )))"; exit 4
fi

LR="${LRS[$lr_idx]}"
# SOFT="${SOFT_ALPHAS[$sa_idx]}"
LAMBDA="${LAMBDAS[$lambda_idx]}"


# echo "=== JOB ${SLURM_JOB_ID:-N/A} on $(hostname) ==="
# echo "Injected: LR=${LR}, SOFT_ALPHA=${SOFT}"
# echo "Fixed: EXPERIMENT=${EXPERIMENT}, BATCH=${BATCH}, GPU=${GPU}"
# echo "Will sweep: ${#LAMBDAS[@]} lambdas × ${#METHODS[@]} methods × ${#REFS[@]} refs"


echo "=== JOB ${SLURM_JOB_ID:-N/A} on $(hostname) ==="
echo "Injected: LR=${LR}, LAMBDA=${LAMBDA}"
echo "Fixed: EXPERIMENT=${EXPERIMENT}, BATCH=${BATCH}, GPU=${GPU}, SOFT_ALPHA=${SOFT_ALPHAS[0]}, METHODS=${METHODS[0]}"
echo "Will sweep: ${#LAMBDAS[@]} lambdas x ${#REFS[@]} refs"
echo "-----------------------------------------------"

mkdir -p logs

JID="${SLURM_JOB_ID:-local$$}"
STAMP="$(date +%Y%m%d-%H%M%S)"

# ---------------- Nested loops (no array) ----------------
for REF in "${REFS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    for SOFT in "${SOFT_ALPHAS[@]}"; do
      REF_BASENAME="$(basename "$REF")"
      TAG="lr${LR}_lam${LAMBDA}_sa${SOFT}_${METHOD}_${REF_BASENAME}"
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
        "optimizer.configs.lr=${LR}"
        "loss.configs.soft_alpha=${SOFT}"
        "loss.configs.lambda_aux=${LAMBDA}"
        "loss.configs.use_token_loss=True"
        "loss.configs.evidence_selection_strategy=reference_model"
        "loss.configs.reference_model_path=${REF}"
        "loss.configs.fallback_to_full_attention_if_empty=False"
        "loss.configs.explanation_method=${METHOD}"
      )

      {
        echo "JOB=${JID}"
        echo "HOST=$(hostname)"
        echo "LR=${LR}"
        echo "lambda_aux=${LAMBDA}"
        echo "soft_alpha=${SOFT}"
        echo "explanation_method=${METHOD}"
        echo "reference_model_path=${REF}"
        echo "[CMD] ${CMD[*]}"
        echo "-------------------------------------------"
      } >"$LOG"

      # Run and append stdout, capture stderr separately
      "${CMD[@]}" >>"$LOG" 2>"$ERR"

      if [[ $? -eq 0 ]]; then
        echo "✅ Done: ${TAG}" | tee -a "$LOG"
      else
        echo "❌ Failed: ${TAG} (see $ERR)" | tee -a "$LOG"
      fi
    done
  done
done

# echo "=== All runs completed for LR=${LR}, SOFT_ALPHA=${SOFT} ==="
echo "=== All runs completed for LR=${LR}, LAMBDA=${LAMBDA} ==="
