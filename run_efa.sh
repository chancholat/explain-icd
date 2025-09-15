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

# ---------------- Allowed options (for index-based inject) ----------------
LRS=(5e-5 1e-5)
METHODS=(laat grad_attention)
REFS=("models/supervised/ym0o7co8" "models/supervised_attention_full_target")

# Fixed singletons
SOFT_ALPHA=0.1
LAMBDA_AUX=0

# ---------------- Helpers ----------------
is_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_float()   { [[ "$1" =~ ^([0-9]*\.)?[0-9]+([eE][-+]?[0-9]+)?$ ]]; }

resolve_lr() {
  local arg="$1"
  if is_integer "$arg"; then
    (( arg < 0 || arg >= ${#LRS[@]} )) && { echo "LR index out of range"; return 1; }
    echo "${LRS[$arg]}"
  elif is_float "$arg"; then
    echo "$arg"
  else
    echo "Invalid LR: '$arg' (use index or float)"; return 1
  fi
}

resolve_method() {
  local arg="$1"
  if is_integer "$arg"; then
    (( arg < 0 || arg >= ${#METHODS[@]} )) && { echo "METHOD index out of range"; return 1; }
    echo "${METHODS[$arg]}"
  else
    for m in "${METHODS[@]}"; do
      [[ "$m" == "$arg" ]] && { echo "$m"; return 0; }
    done
    echo "Unknown METHOD: '$arg' (valid: ${METHODS[*]})"; return 1
  fi
}

resolve_ref() {
  local arg="$1"
  if is_integer "$arg"; then
    (( arg < 0 || arg >= ${#REFS[@]} )) && { echo "REF index out of range"; return 1; }
    echo "${REFS[$arg]}"
  else
    echo "$arg"   # treat as explicit path
  fi
}

usage() {
  cat <<EOF
Usage (inject-only):
  sbatch $0 <LR> <METHOD> <REF>

  <LR>      = index into LRS (${!LRS[@]}) or a float (e.g., 3e-5)
  <METHOD>  = index into METHODS (${!METHODS[@]}) or name (${METHODS[*]})
  <REF>     = index into REFS (${!REFS[@]}) or a path

Examples:
  sbatch $0 0 1 0
  sbatch $0 3e-5 laat   
EOF
}

# ---------------- Parse & resolve ----------------
if [[ $# -ne 3 ]]; then
  usage
  exit 1
fi

LR="$(resolve_lr "$1")"         || { echo "$LR"; exit 2; }
METHOD="$(resolve_method "$2")" || { echo "$METHOD"; exit 3; }
REF="$(resolve_ref "$3")"       || { echo "$REF"; exit 4; }

mkdir -p logs
JID="${SLURM_JOB_ID:-local$$}"
STAMP="$(date +%Y%m%d-%H%M%S)"

echo "=== JOB ${SLURM_JOB_ID:-N/A} on $(hostname) ==="
echo "Injected: LR=${LR}, METHOD=${METHOD}, REF=${REF}"
echo "Fixed: EXPERIMENT=${EXPERIMENT}, BATCH=${BATCH}, GPU=${GPU}"
echo "-----------------------------------------------"

REF_BASENAME="$(basename "$REF")"
TAG="lr${LR}_lam${LAMBDA_AUX}_${METHOD}_${REF_BASENAME}"
LOG="logs/plm_${TAG}_job${JID}_${STAMP}.out"
ERR="logs/plm_${TAG}_job${JID}_${STAMP}.err"

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
  "loss.configs.mask_pooling=False"
  "loss.configs.explanation_method=${METHOD}"
)

{
  echo "JOB=${JID}"
  echo "HOST=$(hostname)"
  echo "LR=${LR}"
  echo "explanation_method=${METHOD}"
  echo "reference_model_path=${REF}"
  echo "[CMD] ${CMD[*]}"
  echo "-------------------------------------------"
} >"$LOG"

"${CMD[@]}" >>"$LOG" 2>"$ERR"
status=$?

if [[ $status -eq 0 ]]; then
  echo "✅ Done: ${TAG}" | tee -a "$LOG"
else
  echo "❌ Failed: ${TAG} (see $ERR)" | tee -a "$LOG"
fi

exit $status
