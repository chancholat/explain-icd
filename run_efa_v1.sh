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
LRS=(5e-5 1e-5)          # $1 -> 0..1
SCALE_FACTORS=(0.1 0.3 1.0) # $2 -> 0..2 

# sweep over window_stride
WINDOW_STRIDES=(0 3 10 30)

# Fixed singletons
REF="models/supervised/ym0o7co8"
SOFT_ALPHA=0
LAMBDA_AUX=0
METHOD=(laat)

lr_idx=$1
sc_factor=$2

if (( lr_idx < 0 || lr_idx >= ${#LRS[@]} )); then
  echo "ERROR: lr_idx out of range (0..$(( ${#LRS[@]}-1 )))"; exit 2
fi
if (( sc_factor < 0 || sc_factor >= ${#SCALE_FACTORS[@]} )); then
  echo "ERROR: sc_factor out of range (0..$(( ${#SCALE_FACTORS[@]}-1 )))"; exit 2
fi

LR="${LRS[$lr_idx]}"
SCALE_FACTOR="${SCALE_FACTORS[$sc_factor]}"

echo "=== JOB ${SLURM_JOB_ID:-N/A} on $(hostname) ==="
echo "Injected: LR=${LR}, SCALE_FACTOR=${SCALE_FACTOR}"
echo "Fixed: EXPERIMENT=${EXPERIMENT}, BATCH=${BATCH}, GPU=${GPU}, REF=${REF}, METHOD=${METHOD}"
echo "Will sweep: window_strides × ${#WINDOW_STRIDES[@]}"
echo "-----------------------------------------------"

mkdir -p logs

JID="${SLURM_JOB_ID:-local$$}"
STAMP="$(date +%Y%m%d-%H%M%S)"

# ---------------- One loop over window_stride ----------------
for WINDOW_STRIDE in "${WINDOW_STRIDES[@]}"; do
  REF_BASENAME="$(basename "$REF")"
  TAG="lr${LR}_${METHOD}_${REF_BASENAME}_ws${WINDOW_STRIDE}_sf${SCALE_FACTOR}"
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
    "loss.configs.explanation_threshold_scale=${SCALE_FACTOR}"
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

echo "=== All runs completed for LR=${LR}, METHOD=${METHOD} ==="