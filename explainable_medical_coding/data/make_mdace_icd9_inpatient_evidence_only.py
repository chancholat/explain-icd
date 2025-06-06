"""
Script to create the mimiciii_full dataset from the mimiciii dataset. The dataset is from the Mullenbach et al. paper.
"""

from pathlib import Path

import numpy as np
import polars as pl
import sys

sys.path.append("./")

from explainable_medical_coding.utils.settings import TEXT_COLUMN


def convert_to_older_icd9_version(
    old_codes: set[str], new_codes: set[str]
) -> dict[str, str]:
    """Converts the old ICD-9 codes to the new ICD-9 codes.

    Args:
        old_codes (set[str]): The old ICD-9 codes.
        new_codes (set[str]): The new ICD-9 codes.

    Returns:
        dict[str, str]: The mapping from old ICD-9 codes to new ICD-9 codes.
    """
    mapping = {}

    for new_code in new_codes:
        if new_code in old_codes:
            mapping[new_code] = new_code
        else:
            temp_new_code = new_code[:-1]
            while True:
                if len(temp_new_code) < 3:
                    print(f"Could not find mapping for {new_code}")
                    break
                if temp_new_code in old_codes:
                    print(f"Found mapping from {new_code} to {temp_new_code}")
                    mapping[new_code] = temp_new_code
                    break
                temp_new_code = temp_new_code[:-1]

    return mapping


new_folder_path = Path("data/processed/mdace_icd9_inpatient_evidence_only")
# create folder
new_folder_path.mkdir(parents=True, exist_ok=True)

notes = pl.read_parquet("data/processed/mdace_notes.parquet")
mimiciv = pl.read_parquet("data/processed/mimiciv.parquet")
annotations = pl.read_parquet("data/processed/mdace_inpatient_annotations.parquet")
annotations_icd9 = annotations.filter(pl.col("code_type").is_in({"icd9cm", "icd9pcs"}))

mimiciv_icd9cm_codes = set(
    mimiciv.filter(pl.col("diagnosis_code_type") == "icd9cm")["diagnosis_codes"]
    .explode()
    .unique()
)
mimiciv_icd9pcs_codes = set(
    mimiciv.filter(pl.col("procedure_code_type") == "icd9pcs")["procedure_codes"]
    .explode()
    .unique()
)

annotations_icd9cm_codes = set(
    annotations_icd9.filter(pl.col("code_type") == "icd9cm")["code"].explode().unique()
)
annotations_icd9pcs_codes = set(
    annotations_icd9.filter(pl.col("code_type") == "icd9pcs")["code"].explode().unique()
)

icd9cm_mapping = convert_to_older_icd9_version(
    mimiciv_icd9cm_codes, annotations_icd9cm_codes
)
icd9pcs_mapping = convert_to_older_icd9_version(
    mimiciv_icd9pcs_codes, annotations_icd9pcs_codes
)
annotations_icd9 = annotations_icd9.with_columns(
    pl.col("code")
    .map_elements(lambda code: icd9cm_mapping.get(code, code))
    .alias("code")
)
annotations_icd9 = annotations_icd9.with_columns(
    pl.col("code")
    .map_elements(lambda code: icd9pcs_mapping.get(code, code))
    .alias("code")
)

annotations_icd9_cm = (
    annotations_icd9.filter(pl.col("code_type") == "icd9cm")
    .group_by(["note_id"])
    .agg(
        pl.col("code").map_elements(list).alias("diagnosis_codes"),
        pl.col("spans").map_elements(list).alias("diagnosis_code_spans"),
        pl.col("code_type").last().alias("diagnosis_code_type"),
    )
)
annotations_icd9_pcs = (
    annotations_icd9.filter(pl.col("code_type") == "icd9pcs")
    .group_by(["note_id"])
    .agg(
        pl.col("code").map_elements(list).alias("procedure_codes"),
        pl.col("spans").map_elements(list).alias("procedure_code_spans"),
        pl.col("code_type").last().alias("procedure_code_type"),
    )
)

annotations_icd9 = annotations_icd9_cm.join(
    annotations_icd9_pcs, on="note_id", how="outer_coalesce"
)


data = notes.join(annotations_icd9, on="note_id")
data = data.with_columns(
    [
        pl.col("diagnosis_codes").fill_null([]),
        pl.col("procedure_codes").fill_null([]),
        pl.col("diagnosis_code_spans").fill_null([]),
        pl.col("procedure_code_spans").fill_null([]),
        pl.col("diagnosis_code_type").fill_null("icd9cm"),
        pl.col("procedure_code_type").fill_null("icd9pcs"),
    ]
)

data = data.with_columns(
    pl.concat_list(["diagnosis_codes", "procedure_codes"]).alias("codes")
)
data = data.with_columns(
    pl.concat_list(["diagnosis_code_spans", "procedure_code_spans"]).alias(
        "target_spans"
    )
)

data = data.explode(["target_spans", "codes"])
data = data.with_columns(
    pl.struct(TEXT_COLUMN, "target_spans")
    .map_elements(
        lambda row: " ".join(
            [row[TEXT_COLUMN][span[0] : span[1] + 1] for span in row["target_spans"]]
        )
    )
    .alias(TEXT_COLUMN)
)
data = data.drop(
    [
        "CHARTDATE",
        "CHARTTIME",
        "STORETIME",
        "CGID",
        "ISERROR",
        "diagnosis_codes",
        "procedure_codes",
        "diagnosis_code_spans",
        "procedure_code_spans",
        "diagnosis_code_type",
        "procedure_code_type",
        "target_spans",
    ]
)
data = data.with_columns(evidence_id=np.arange(len(data)))
data = data.with_columns(pl.col("codes").map_elements(lambda x: [x]))
train_split = pl.read_csv(
    "data/splits/mdace/inpatient/MDace-ev-train.csv",
    has_header=False,
    new_columns=["_id"],
)
val_split = pl.read_csv(
    "data/splits/mdace/inpatient/MDace-ev-val.csv",
    has_header=False,
    new_columns=["_id"],
)
test_split = pl.read_csv(
    "data/splits/mdace/inpatient/MDace-ev-test.csv",
    has_header=False,
    new_columns=["_id"],
)

train = data.filter(pl.col("_id").is_in(train_split["_id"]))
val = data.filter(pl.col("_id").is_in(val_split["_id"]))
test = data.filter(pl.col("_id").is_in(test_split["_id"]))

train.write_parquet(new_folder_path / "train.parquet")
val.write_parquet(new_folder_path / "val.parquet")
test.write_parquet(new_folder_path / "test.parquet")
