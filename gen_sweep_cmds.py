import itertools

# --- Fixed parts ---
BASE_CMD = [
    "python train_plm.py",
    "experiment=mdace_icd9_code/plm_efa",
    "gpu=0",
    "loss.configs.evidence_selection_strategy=reference_model",
    "loss.configs.fallback_to_full_attention_if_empty=False",
]

# config for A100 40GB RAM
MAX_BATCH_SIZE = 8
BATCH_SIZE = 8

# --- Sweeps ---
LRS = [5e-5, 1e-5]
LAMBDA_AUX = [1.0, 0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001, 0.000001]
# We could run sweep the lambda_aux and lr first to find the best lr and lambda_aux, then fixed them and run with other explanation methods and reference models and soft alpha.

# ---- Fixed for first experiment ----
EXPLANATION_METHODS = ["laat"]
REFERENCE_MODEL_PATHS = [
    "models/supervised/ym0o7co8",
]
SOFT_ALPHAS = [0]

# ------- Other methods and reference models to try --------: 
# EXPLANATION_METHODS ["grad_attention"]
# REFERENCE_MODEL_PATHS [
#     "models/suppervised_attention_2",
#     "models/plm_icd",
#     "models/pl_igr",
#     "models/plm_pgd"
# ]
# SOFT_ALPHAS [0.01, 0.1, 0.5, 1, 2, 5]


# --- Cartesian product ---
jobs = itertools.product(LRS, LAMBDA_AUX,
                         EXPLANATION_METHODS, REFERENCE_MODEL_PATHS, SOFT_ALPHAS)

for lr, lam, method, ref, alpha in jobs:
    cmd = BASE_CMD + [
        f"dataloader.max_batch_size={MAX_BATCH_SIZE}",
        f"dataloader.batch_size={BATCH_SIZE}",
        f"optimizer.configs.lr={lr}",
        f"loss.configs.lambda_aux={lam}",
        f"loss.configs.explanation_method={method}",
        f"loss.configs.reference_model_path={ref}",
        f"loss.configs.soft_alpha={alpha}",
        "loss.configs.use_token_loss=True",
    ]
    print(" ".join(cmd))

# Run the following command in terminal to generate sweep commands and save to a shell script file:
# python gen_sweep_cmds.py > sweep_cmds.sh

# Then run the shell script file:
# bash sweep_cmds.sh
