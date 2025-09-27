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
MAX_BATCH=16
BATCH=16
GPU=0

# ---------------- Grids ----------------
LRS=(5e-5 1e-5)                          # $1 -> 0..1
REFS=("models/supervised/ym0o7co8" "models/supervised_attention_full_target")  # $3 -> 0..1
WINDOW_STRIDES=(6 10 15 20)

# Fixed singletons
SOFT_ALPHA=0
METHOD="laat"
LAMBDA_AUX=0

# ---------------- Parse CLI args ----------------
if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch $0 <lr_idx:0-1> <method_idx:0-1> <ref_idx:0-1>"
  exit 1
fi

lr_idx=$1
ref_idx=$2

# Basic bounds checks
if (( lr_idx < 0 || lr_idx >= ${#LRS[@]} )); then
  echo "ERROR: lr_idx out of range (0..$(( ${#LRS[@]}-1 )))"; exit 2
fi
if (( ref_idx < 0 || ref_idx >= ${#REFS[@]} )); then
  echo "ERROR: ref_idx out of range (0..$(( ${#REFS[@]}-1 )))"; exit 4
fi

LR="${LRS[$lr_idx]}"
REF="${REFS[$ref_idx]}"

echo "=== JOB ${SLURM_JOB_ID:-N/A} on $(hostname) ==="
echo "Injected: LR=${LR}, REF=${REF}"
echo "Fixed: EXPERIMENT=${EXPERIMENT}, BATCH=${BATCH}, GPU=${GPU}"
echo "Will sweep: window_strides × ${#WINDOW_STRIDES[@]}"
echo "-----------------------------------------------"

mkdir -p logs

JID="${SLURM_JOB_ID:-local$$}"
STAMP="$(date +%Y%m%d-%H%M%S)"

# ---------------- One loop over window_stride ----------------
for WINDOW_STRIDE in "${WINDOW_STRIDES[@]}"; do
  REF_BASENAME="$(basename "$REF")"
  TAG="lr${LR}_sa${SOFT_ALPHA}_${METHOD}_${REF_BASENAME}_ws${WINDOW_STRIDE}"
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
    "loss.configs.soft_alpha=${SOFT_ALPHA}"
    "loss.configs.lambda_aux=${LAMBDA_AUX}"
    "loss.configs.use_token_loss=False"
    "loss.configs.evidence_selection_strategy=reference_model"
    "loss.configs.reference_model_path=${REF}"
    "loss.configs.fallback_to_full_attention_if_empty=False"
    "loss.configs.explanation_method=${METHOD}"
    "loss.configs.mask_pooling=False"
    "loss.configs.window_stride=${WINDOW_STRIDE}"
  )

  {
    echo "JOB=${JID}"
    echo "HOST=$(hostname)"
    echo "LR=${LR}"
    echo "explanation_method=${METHOD}"
    echo "reference_model_path=${REF}"
    echo "window_stride=${WINDOW_STRIDE}"
    echo "[CMD] ${CMD[*]}"
    echo "-------------------------------------------"
  } >"$LOG"

  "${CMD[@]}" >>"$LOG" 2>"$ERR"

  if [[ $? -eq 0 ]]; then
    echo "✅ Done: ${TAG}" | tee -a "$LOG"
  else
    echo "❌ Failed: ${TAG} (see $ERR)" | tee -a "$LOG"
  fi
done

echo "=== All runs completed for LR=${LR}, METHOD=${METHOD}, REF=${REF} ==="
