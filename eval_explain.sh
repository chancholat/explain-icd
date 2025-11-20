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
export WANDB_API_KEY=c76d20783a5b6c0eb844caaf78d65aef0e27d699

# ---------------- Fixed base config ----------------
GPU=0

mkdir -p logs
JID="${SLURM_JOB_ID:-local$$}"
STAMP="$(date +%Y%m%d-%H%M%S)"

RUN_IDS=(v99we365 8wpisgnm m14ghn5j pgd/zauungyt)

for RID in "${RUN_IDS[@]}"; do
  LOG="logs/eval_explain_${RID}_job${JID}_${STAMP}.out"
  ERR="logs/eval_explain_${RID}_job${JID}_${STAMP}.err"

  echo ">> RUN run_id=${RID}"
  echo "   -> log: ${LOG}"
  echo "   -> err: ${ERR}"

  python eval_explanations.py \
    "gpu=${GPU}" \
    "run_id=${RID}" \
    "model_name=baseline" \
    "explainers=[laat]" \
    "evaluate_faithfulness=True" \
    >>"$LOG" 2>"$ERR"

  if [[ $? -eq 0 ]]; then
    echo "✅ Done run_id=${RID}" | tee -a "$LOG"
  else
    echo "❌ Failed run_id=${RID} (see $ERR)" | tee -a "$LOG"
  fi
done
