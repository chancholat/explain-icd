#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --qos=priority
#SBATCH --mail-type=END,TIME_LIMIT
#SBATCH --mail-user=thin.nguyen@deakin.edu.au
#SBATCH --array=0-10

# --- Environment setup ---
module load Anaconda3
source activate
conda activate haichan
export PYTHONNOUSERSITE=1
export WANDB_API_KEY=c76d20783a5b6c0eb844caaf78d65aef0e27d699


lambda_idx=$SLURM_ARRAY_TASK_ID

LAMBDA_AUXs=(1.0 0 0.5 0.1 0.05 0.01 0.005 0.001 0.0001 0.00001 0.000001)
LAMBDA_AUX=${LAMBDA_AUXs[$((model_name_idx - 1))]}

# --- Input arguments ---
model_name_idx=$1   # ("gpt2-large" "gpt2-xl" "pythia28")
dataset_idx=$2      # 1=hh, 2=shp
batch_size=64
# run_sft=$4          # 1=true, 0=false
run_irred=$3          # 1=true, 0=false
variant=$SLURM_ARRAY_TASK_ID          

gradient_accumulation_steps=$((batch_size / 4))


if [ -z "$variant" ] || [ -z "$dataset_idx" ]; then
    echo "❌ Usage: sbatch run_2steps.sh <variant: 1=dpo, 2=...> <dataset_idx: 1=hh, 2=shp>"
    exit 1
fi

# --- Model config (fixed) ---
model_names=("gpt2-large" "gpt2-xl" "pythia28")
model_name="${model_names[$((model_name_idx - 1))]}"
if [ ! -f "config/model/${model_name}.yaml" ]; then
    echo "❌ Model config not found: config/model/${model_name}.yaml"
    exit 1
fi

# --- Dataset mapping ---
datasets=("hh" "shp")
if ((dataset_idx < 1 || dataset_idx > 2)); then
    echo "❌ Invalid dataset index: $dataset_idx (must be 1-2)"
    exit 1
fi
dataset="${datasets[$((dataset_idx - 1))]}"
# --- Loss variant ---
case "$variant" in
    1)
        run_top_m="irreducible.top_m=0.5"
        variant_name="dpo0.5"
        ;;
    2)
        run_top_m="irreducible.top_m=0.7"
        variant_name="dpo0.7"

        ;;
   
    3)
        run_top_m="irreducible.top_m=0.8"
        variant_name="dpo0.8"
        ;;
        
    4)
        run_top_m="irreducible.top_m=0.9"
        variant_name="dpo0.9"
        ;;
        

    
    
    *)

        echo "❌ Invalid variant: $variant (must be 1–4)"
        exit 1
        ;;
esac



# --- Logging ---
log_dir="logs"
mkdir -p "$log_dir"
log_file="${log_dir}/${model_name}_${dataset}_${variant_name}_${SLURM_JOB_ID}_${model_name_idx}_${dataset_idx}_${variant}_${batch_size}.out"
err_file="${log_dir}/${model_name}_${dataset}_${variant_name}_${SLURM_JOB_ID}_${model_name_idx}_${dataset_idx}_${variant}_${batch_size}.err"
exec > "$log_file" 2> "$err_file"

echo "🚀 Starting $variant_name on dataset=$dataset with model=$model_name"
echo "-------------------------------------------------------------"



#-----Calculate Irreducible loss ----
if [ "$run_irred" = "1" ]; then
    echo " Running Irreducible..."
    python compute_irreducible.py \
        model=$model_name \
        datasets=[$dataset] \
        batch_size=$batch_size \
        exp_name=${dataset}_${model_name}_irred
fi 

#----Find the Irreducible loss file path ----
BASE_DIR=".cache/thinng"
PREFIX=${dataset}_${model_name}_irred 
latest_suffix=$(find "$BASE_DIR" -maxdepth 1 -type d -name "${PREFIX}*" | \
  sed -E "s|.*/${PREFIX}||" | \
  sort | \
  tail -n 1)

if [ -n "$latest_suffix" ]; then
  irred_path="$BASE_DIR/${PREFIX}${latest_suffix}/irred_loss.pt"
  echo "✅ Found irreducible loss file: $irred_path"
else
  echo "❌ No irreducible loss file found."
  exit 1
fi




# --- Find the latest SFT checkpoint ---

PREFIX="${dataset}_${model_name}_sft_"
latest_suffix=$(find "$BASE_DIR" -maxdepth 1 -type d -name "${PREFIX}*" | \
  sed -E "s|.*/${PREFIX}||" | \
  sort | \
  tail -n 1)

if [ -n "$latest_suffix" ]; then
  ckpt_path="$BASE_DIR/${PREFIX}${latest_suffix}/LATEST/policy.pt"
  echo "✅ Found latest checkpoint: $ckpt_path"
else
  echo "❌ No matching checkpoints found."
  exit 1
fi



# --- Subset selection DPO ---
echo "🔥 Running $variant_name..."
python -u train.py \
    model=$model_name \
    datasets=[$dataset] \
    loss=dpo \
    $run_top_m \
    irreducible.path=$irred_path \
    loss.beta=0.1 \
    wandb.enabled=true \
    exp_name=${dataset}_${model_name}_${variant_name} \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    batch_size=$batch_size \
    eval_batch_size=$batch_size \
    trainer=BasicTrainer \
    sample_during_eval=false \
    model.archive=$ckpt_path

echo "✅ Done: $variant_name on $dataset"


