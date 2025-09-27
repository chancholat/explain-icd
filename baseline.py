# ruff: noqa: E402
import logging
import math
from pathlib import Path

# load environment variables
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import hydra
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import explainable_medical_coding.config.factories as factories
from explainable_medical_coding.utils.loaders import (
    load_and_prepare_dataset,
    load_trained_model,
)
from explainable_medical_coding.utils.tokenizer import TargetTokenizer
from explainable_medical_coding.utils.data_helper_functions import (
    create_targets_column,
    filter_unknown_targets,
    format_evidence_spans,
    get_unique_targets,
)
from explainable_medical_coding.utils.seed import set_seed
from explainable_medical_coding.utils.settings import TARGET_COLUMN, TEXT_COLUMN
from explainable_medical_coding.utils.tensor import deterministic, set_gpu

LOGGER = logging.getLogger(name=__file__)
LOGGER.setLevel(logging.INFO)


@hydra.main(
    version_base=None,
    config_path="explainable_medical_coding/config",
    config_name="config",
)
def main(cfg: OmegaConf) -> None:
    if cfg.deterministic:
        deterministic()

    set_seed(cfg.seed)
    device = set_gpu(cfg)

    # Check if model path is provided
    if cfg.load_model is None:
        raise ValueError("Model path must be provided in config.load_model for baseline evaluation")
    
    model_path = Path(cfg.load_model)
    target_columns = list(cfg.data.target_columns)
    dataset_path = Path(cfg.data.dataset_path)
    
    LOGGER.info(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(str(dataset_path), trust_remote_code=True)

    # Load text tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.configs.model_path,
    )
    max_input_length = int(cfg.data.max_length)
    
    # Tokenize text
    dataset = dataset.map(
        lambda x: text_tokenizer(
            x[TEXT_COLUMN],
            return_length=True,
            truncation=True,
            max_length=max_input_length,
        ),
        batched=True,
        num_proc=8,
        batch_size=1_000,
        desc="Tokenizing text",
    )

    # Create targets column
    dataset = dataset.map(
        lambda x: create_targets_column(x, target_columns),
        desc="Creating targets column",
    )
    
    # Load target tokenizer from saved model
    LOGGER.info(f"Loading target tokenizer from: {model_path}")
    autoregressive = bool(cfg.model.autoregressive)
    target_tokenizer = TargetTokenizer(autoregressive=autoregressive)
    target_tokenizer.load(model_path / "target_tokenizer.json")

    # Filter unknown targets and empty targets
    # known_targets = set(target_tokenizer.target2id.keys())
    known_targets = set(get_unique_targets(dataset))
    dataset = dataset.map(
        lambda x: filter_unknown_targets(x, known_targets=known_targets),
        desc="Filter unknown targets",
    )
    dataset = dataset.filter(
        lambda x: len(x[TARGET_COLUMN]) > 0, desc="Filtering empty targets"
    )
    
    # Format evidence spans and convert targets to IDs
    dataset = dataset.map(lambda x: format_evidence_spans(x, text_tokenizer))
    dataset = dataset.map(
        lambda x: {"target_ids": target_tokenizer(x[TARGET_COLUMN])},
        desc="Converting targets to target ids",
    )
    
    # Set format for PyTorch
    dataset.set_format(
        type="torch", columns=["input_ids", "length", "attention_mask", "target_ids"]
    )

    # Create lookups
    lookups = factories.get_lookups(
        dataset=dataset,
        text_tokenizer=text_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    LOGGER.info(lookups.data_info)

    # Load trained model
    LOGGER.info(f"Loading trained model from: {model_path}")
    saved_config = OmegaConf.load(model_path / "config.yaml")
    model, decision_boundary = load_trained_model(
        model_path,
        saved_config,
        pad_token_id=text_tokenizer.pad_token_id,
        device=device,
    )
    model.to(device)
    model.eval()
    
    LOGGER.info(f"Loaded model with decision boundary: {decision_boundary}")

    # Create loss function
    loss_function = factories.get_loss_function(config=cfg.loss)

    # Create dataloaders
    dataloaders = factories.get_dataloaders(
        config=cfg.dataloader,
        dataset=dataset,
        target_tokenizer=lookups.target_tokenizer,
        pad_token_id=lookups.data_info["pad_token_id"],
    )

    # Create metric collections
    metric_collections = factories.get_metric_collections(
        config=cfg.metrics,
        number_of_classes=lookups.data_info["num_classes"],
        split2code_indices=lookups.split2code_indices,
        autoregressive=cfg.model.autoregressive,
    )

    # Create dummy optimizer (required for trainer initialization but not used)
    optimizer = factories.get_optimizer(config=cfg.optimizer, model=model)
    
    # Create callbacks
    callbacks = factories.get_callbacks(config=cfg.callbacks)
    
    # Create trainer
    trainer_class = factories.get_trainer(name=cfg.trainer.name)
    trainer = trainer_class(
        config=cfg,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        dataloaders=dataloaders,
        metric_collections=metric_collections,
        callbacks=callbacks,
        lr_scheduler=None,  # No scheduler needed for evaluation
        lookups=lookups,
        accumulate_grad_batches=1,  # Not relevant for evaluation
    ).to(device)

    # Set the pre-trained decision boundary
    if hasattr(trainer, 'best_db'):
        trainer.best_db = decision_boundary
    
    # Set threshold for test metrics if threshold tuning is enabled
    if cfg.trainer.threshold_tuning:
        trainer.metric_collections["test"].set_threshold(decision_boundary)

    LOGGER.info("Starting evaluation on validation set...")
    trainer.val(epoch=0, split_name="validation", evaluating_best_model=True)
    
    LOGGER.info("Starting evaluation on test set...")
    trainer.val(epoch=0, split_name="test", evaluating_best_model=True)
    
    LOGGER.info("Baseline evaluation completed!")


if __name__ == "__main__":
    main()