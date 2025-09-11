#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --qos=priority
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=thin.nguyen@deakin.edu.au
#SBATCH --array=0-10            # 11 values below → indices 0..10

# ------------- Environment -------------
module load Anaconda3
source activate
conda activate haichan
export PYTHONNOUSERSITE=1
export WANDB_API_KEY=c76d20783a5b6c0eb844caaf78d65aef0e27d699  # <- consider using a secret manager

# ------------- Fixed base config -------------
EXPERIMENT="mdace_icd9/plm_efa"
MAX_BATCH=8
BATCH=8
GPU=0

# ------------- Grids -------------
# Only lambda uses the SLURM array index
LAMBDAS=(1.0 0 0.5 0.1 0.05 0.01 0.005 0.001 0.0001 0.00001 0.000001)
lambda_idx="${SLURM_ARRAY_TASK_ID}"
LAMBDA="${LAMBDAS[$lambda_idx]}"

# The rest are injected via CLI indices ($1..$4)
LRS=(5e-5 1e-5)                                     # $1
SOFT_ALPHAS=(0.5 0.3 0.1 0.01)                      # $2
METHODS=(laat grad_attention)                       # $3
REFS=("models/supervised/ym0o7co8" "models/suppervised_attention_2")  # $4

# ------------- Parse CLI args -------------
if [[ $# -lt 4 ]]; then
  echo "Usage: sbatch $0 <lr_idx:0-1> <soft_alpha_idx:0-3> <method_idx:0-1> <ref_idx:0-1>"
  echo "       (lambda index comes from --array=${SLURM_ARRAY_TASK_ID})"
  exit 1
fi

lr_idx="$1"
sa_idx="$2"
method_idx="$3"
ref_idx="$4"


LR="${LRS[$lr_idx]}"
SOFT="${SOFT_ALPHAS[$sa_idx]}"
METHOD="${METHODS[$method_idx]}"
REF="${REFS[$ref_idx]}"

# ------------- Logging -------------
mkdir -p logs
TAG="lr${LR}_lam${LAMBDA}_sa${SOFT}_${METHOD}_$(basename "$REF")"
LOG="logs/plm_${TAG}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.out"
ERR="logs/plm_${TAG}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.err"
exec >"$LOG" 2>"$ERR"

echo "JOB=${SLURM_JOB_ID} TASK=${SLURM_ARRAY_TASK_ID}"
echo "LR=${LR}"
echo "lambda_aux=${LAMBDA}"
echo "soft_alpha=${SOFT}"
echo "explanation_method=${METHOD}"
echo "reference_model_path=${REF}"
echo "-------------------------------------------"

# ------------- Command -------------
CMD=( python train_plm.py
  "experiment=${EXPERIMENT}"
  "dataloader.max_batch_size=${MAX_BATCH}"
  "dataloader.batch_size=${BATCH}"
  "gpu=${GPU}"
  "optimizer.configs.lr=${LR}"
  "loss.configs.soft_alpha=${SOFT}"
  "loss.configs.lambda_aux=${LAMBDA}"
  "loss.configs.use_token_loss=True"
  "loss.configs.mask_pooling=False"
  "loss.configs.evidence_selection_strategy=reference_model"
  "loss.configs.reference_model_path=${REF}"
  "loss.configs.fallback_to_full_attention_if_empty=False"
  "loss.configs.explanation_method=${METHOD}"
)

echo "[CMD] ${CMD[*]}"
"${CMD[@]}"

echo "✅ Done: ${TAG}"
