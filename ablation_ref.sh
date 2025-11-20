#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=120:00:00
#SBATCH --mem=256G
#SBATCH --qos=batch-short
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
REFS=("models/pgd/zauungyt" "models/unsupervised/vxrn54op" "models/supervised_attention_full_target")  # $1 -> 0..2

# Fixed singletons
SCALE_FACTOR=(0.01) 
WINDOW_STRIDE=(10)
SOFT_ALPHA=0
LAMBDA_AUX=0
LR=(5e-5)          
METHOD=(laat) 

ref_idx=$1

if (( ref_idx < 0 || ref_idx >= ${#REFS[@]} )); then
  echo "ERROR: ref_idx out of range (0..$(( ${#REFS[@]}-1 )))"; exit 2
fi

REF="${REFS[$ref_idx]}"

echo "=== JOB ${SLURM_JOB_ID:-N/A} on $(hostname) ==="
echo "Injected: REF=${REF}"
echo "Fixed: EXPERIMENT=${EXPERIMENT}, BATCH=${BATCH}, GPU=${GPU}, REF=${REF}, METHOD=${METHOD}, LR=${LR}, WINDOW_STRIDE=${WINDOW_STRIDE}, SCALE_FACTOR=${SCALE_FACTOR}"
echo "-----------------------------------------------"

mkdir -p logs

JID="${SLURM_JOB_ID:-local$$}"
STAMP="$(date +%Y%m%d-%H%M%S)"


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

echo "=== Run completed for REF=${REF} ==="