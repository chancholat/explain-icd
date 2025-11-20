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
GPU=0

echo "=== JOB ${SLURM_JOB_ID:-N/A} on $(hostname) ==="
echo "-----------------------------------------------"

mkdir -p logs

JID="${SLURM_JOB_ID:-local$$}"
STAMP="$(date +%Y%m%d-%H%M%S)"

LOG="logs/eval_explain_job${JID}_${STAMP}.out"
ERR="logs/eval_explain_job${JID}_${STAMP}.err"

echo ""
echo ">> RUN"
echo "   -> log: ${LOG}"
echo "   -> err: ${ERR}"

CMD=( python eval_explanations.py
"gpu=${GPU}"
"--multirun"
"run_id=v99we365,8wpisgnm,m14ghn5j"
"model_name=baseline"
"explainers=[laat]"
"evaluate_faithfulness=True"
)

{
echo "JOB=${JID}"
echo "HOST=$(hostname)"
echo "-------------------------------------------"
} >"$LOG"

"${CMD[@]}" >>"$LOG" 2>"$ERR"

if [[ $? -eq 0 ]]; then
echo "✅ Done" | tee -a "$LOG"
else
echo "❌ Failed (see $ERR)" | tee -a "$LOG"
fi
