# Config for A100 40GB RAM
MAX_BATCH_SIZE = 8
BATCH_SIZE = 8

BASELINE_CMD = [
    "python train_plm.py",
    "gpu=0",
]

for experiment in ["mdace_icd9_code/plm_icd", "mdace_icd9_code/plm_icd_supervised", "mdace_icd9_code/plm_icd_igr", "mdace_icd9_code/plm_icd_pgd"]:
    cmd = BASELINE_CMD + [
        f"dataloader.max_batch_size={MAX_BATCH_SIZE}",
        f"dataloader.batch_size={BATCH_SIZE}",
        f"optimizer.configs.lr={5e-5}",
    ]
    print(" ".join(cmd))