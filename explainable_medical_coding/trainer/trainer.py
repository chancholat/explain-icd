import gc
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable, Optional
from transformers import AutoTokenizer

import pandas as pd
import torch
from omegaconf import OmegaConf
from rich.pretty import pprint
from rich.progress import track
from torch.utils.data import DataLoader
import sys

sys.path.append("./")
from explainable_medical_coding.eval.metrics import MetricCollection
from explainable_medical_coding.trainer.callbacks import BaseCallback
from explainable_medical_coding.utils.datatypes import Lookups
from explainable_medical_coding.utils.decision_boundary import f1_score_db_tuning
from explainable_medical_coding.utils.settings import ID_COLUMN, TARGET_COLUMN
from explainable_medical_coding.utils.analysis import get_explanations
from explainable_medical_coding.config.factories import get_explainability_method
from explainable_medical_coding.eval.plausibility_metrics import find_explanation_decision_boundary


class Trainer:
    def __init__(
        self,
        config: OmegaConf,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloaders: dict[str, DataLoader],
        metric_collections: dict[str, MetricCollection],
        callbacks: list[BaseCallback],
        lookups: Lookups,
        loss_function: Callable,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        accumulate_grad_batches: int = 1,
    ) -> None:
        self.config = config
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.callbacks = callbacks
        self.device = "cpu"
        self.metric_collections = metric_collections
        self.lr_scheduler = lr_scheduler
        self.lookups = lookups
        self.accumulate_grad_batches = accumulate_grad_batches
        pprint(f"Accumulating gradients over {self.accumulate_grad_batches} batch(es).")
        self.validate_on_training_data = config.trainer.validate_on_training_data
        self.print_metrics = config.trainer.print_metrics
        self.epochs = config.trainer.epochs
        self.epoch = 0
        self.use_amp = config.trainer.use_amp
        self.threshold_tuning = config.trainer.threshold_tuning
        self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.experiment_path = Path(mkdtemp())
        self.current_val_results: dict[str, dict[str, torch.Tensor]] = {}
        self.stop_training = False
        self.best_db = 0.5
        self.explanation_decision_boundary = None
        self.reference_model = None
        
        # Load reference model if specified in config
        print(config.loss)
        if hasattr(config.loss.configs, 'reference_model_path') and config.loss.configs.reference_model_path:
            self.load_reference_model(config.loss.configs.reference_model_path)
        
        # Add wrapper for loss function to include common parameters
        self._original_loss_function = self.loss_function
        self.loss_function = self._wrapped_loss_function
        
        self.on_initialisation_end()
        
    def _wrapped_loss_function(self, batch, model, **kwargs):
        """Wrapper for loss function to include common parameters."""
        # Check if we're using masked_pooling_aux_loss
        is_masked_pooling = (self.config.loss.name == 'masked_pooling_aux_loss')
        
        if is_masked_pooling:
            # Add use_token_loss from config if not explicitly provided
            if 'use_token_loss' not in kwargs and hasattr(self.config.loss, 'configs') and hasattr(self.config.loss.configs, 'use_token_loss'):
                kwargs['use_token_loss'] = self.config.loss.configs.use_token_loss
            
            # Add explanation_decision_boundary if not explicitly provided
            if 'explanation_decision_boundary' not in kwargs and self.explanation_decision_boundary is not None:
                kwargs['explanation_decision_boundary'] = self.explanation_decision_boundary
                
            # Add reference_model if available
            if self.reference_model is not None:
                kwargs['reference_model'] = self.reference_model
                
            # Add evidence_selection_strategy from config
            if hasattr(self.config.loss, 'configs') and hasattr(self.config.loss.configs, 'evidence_selection_strategy'):
                kwargs['evidence_selection_strategy'] = self.config.loss.configs.evidence_selection_strategy
                
                # Check configuration coherence
                strategy = self.config.loss.configs.evidence_selection_strategy
                if strategy == "reference_model" and self.reference_model is None:
                    pprint("WARNING: evidence_selection_strategy is set to 'reference_model' but no reference model is provided.")
                    pprint("Falling back to 'auto' strategy.")
                    kwargs['evidence_selection_strategy'] = "auto"
                elif strategy == "training_model":
                    # Ensure we're using the training model - this is already handled by default
                    pprint("Using training model for token attributions as specified by evidence_selection_strategy")
            
            # Add explanation_method from config
            if hasattr(self.config.loss, 'configs') and hasattr(self.config.loss.configs, 'explanation_method'):
                kwargs['explanation_method'] = self.config.loss.configs.explanation_method
        
        return self._original_loss_function(batch, model=model, **kwargs)
        
    def load_reference_model(self, reference_model_path: str) -> None:
        """Load a reference model to compute explanation decision boundary.
        
        Args:
            reference_model_path (str): Path to the reference model checkpoint
        """
        from explainable_medical_coding.utils.loaders import load_trained_model
        
        pprint(f"Loading reference model from {reference_model_path}")
        try:
            model_path = Path(reference_model_path)
            if not model_path.exists():
                pprint(f"Reference model path does not exist: {model_path}")
                return
                
            saved_config = OmegaConf.load(model_path / "config.yaml")
            text_tokenizer = AutoTokenizer.from_pretrained(
                saved_config.model.configs.model_path,
            )
            self.reference_model, _ = load_trained_model(
                model_path,
                saved_config,
                pad_token_id=text_tokenizer.pad_token_id,
                device=self.device,
            )
            self.reference_model.eval()  # Ensure model is in evaluation mode
            pprint("Reference model loaded successfully")
        except Exception as e:
            pprint(f"Error loading reference model: {e}")
            self.reference_model = None

    def calculate_explanation_decision_boundary(self) -> float:
        """Calculate the explanation decision boundary using the validation dataset.
        This function should be called once during initialization and the result will be cached.
        It uses the reference model if available, otherwise uses the training model.
        The calculation respects the evidence_selection_strategy configuration.

        Returns:
            float: The calculated explanation decision boundary
        """
        if self.explanation_decision_boundary is not None:
            return self.explanation_decision_boundary

        # Determine which model to use based on evidence_selection_strategy
        model_to_use = self.model
        strategy = "auto"
        
        if hasattr(self.config.loss, 'configs') and hasattr(self.config.loss.configs, 'evidence_selection_strategy'):
            strategy = self.config.loss.configs.evidence_selection_strategy
            
        if strategy == "reference_model":
            if self.reference_model is not None:
                model_to_use = self.reference_model
                pprint("Using reference model for explanation decision boundary calculation")
            else:
                pprint("WARNING: evidence_selection_strategy is set to 'reference_model' but no reference model is provided.")
                pprint("Falling back to training model for explanation decision boundary calculation")
        elif strategy == "training_model":
            model_to_use = self.model
            pprint("Using training model for explanation decision boundary calculation")
        elif strategy == "auto":
            # In auto mode, prefer reference model if available
            if self.reference_model is not None:
                model_to_use = self.reference_model
                pprint("Using reference model for explanation decision boundary calculation (auto mode)")
            else:
                pprint("Using training model for explanation decision boundary calculation (auto mode)")
        elif strategy == "evidence_ids":
            pprint("Using evidence_ids strategy - explanation decision boundary calculation not needed")
            self.explanation_decision_boundary = 0.05  # Default value, won't be used
            return self.explanation_decision_boundary

        pprint("Calculating explanation decision boundary...")
        try:            
            # Use the full validation dataset
            validation_dataset = self.dataloaders["validation"].dataset
            
            # Get the explainer based on the configured explanation method
            explanation_method = "laat"  # Default method
            if hasattr(self.config.loss, 'configs') and hasattr(self.config.loss.configs, 'explanation_method'):
                explanation_method = self.config.loss.configs.explanation_method
                
            # Map some method names if they differ from factory keys
            method_mapping = {
                "gradient_attention": "grad_attention",
                "integrated_gradients": "integrated_gradient"
            }
            method_key = method_mapping.get(explanation_method, explanation_method)
            
            try:
                # Get the explainer function from the factory
                explainer = get_explainability_method(method_key)
                explainer_callable = explainer(model=model_to_use)
            except ValueError:
                # If the method isn't found, default to LAAT
                print(f"Explanation method '{explanation_method}' not found. Defaulting to LAAT.")
                explainer = get_explainability_method("laat")
                explainer_callable = explainer(model=model_to_use)
            
            # Get explanations for the validation dataset
            explanations_val_df = get_explanations(
                model=model_to_use,
                model_path=self.experiment_path,
                dataset=validation_dataset,
                explainer=explainer_callable,
                target_tokenizer=self.lookups.target_tokenizer,
                cache=True,
                cache_path=Path(".cache")
            )
            
            # Find the explanation decision boundary
            explanation_decision_boundary = find_explanation_decision_boundary(explanations_val_df)
            self.explanation_decision_boundary = explanation_decision_boundary
            
            pprint(f"Explanation decision boundary calculated: {explanation_decision_boundary}")
            return explanation_decision_boundary
        except Exception as e:
            pprint(f"Error calculating explanation decision boundary: {e}")
            # Fallback to a default value if calculation fails
            self.explanation_decision_boundary = 0.05
            return 0.05

    def fit(self) -> None:
        """Train and validate the model."""
        try:
            self.save_configs()
            self.on_fit_begin()
            
            # Check if we're using masked_pooling_aux_loss
            # is_masked_pooling = (self._original_loss_function.__name__ == 'masked_pooling_aux_loss' or 
            #                     self.config.loss.name == 'masked_pooling_aux_loss')
            is_masked_pooling = (self.config.loss.name == 'masked_pooling_aux_loss')
            
            # Calculate the explanation decision boundary before training if needed
            if is_masked_pooling:
                # Check the evidence selection strategy
                need_threshold = True
                if hasattr(self.config.loss, 'configs') and hasattr(self.config.loss.configs, 'evidence_selection_strategy'):
                    strategy = self.config.loss.configs.evidence_selection_strategy
                    # Only calculate threshold if needed based on strategy
                    if strategy == "evidence_ids":
                        need_threshold = False
                
                if need_threshold:
                    self.calculate_explanation_decision_boundary()
            
            for _ in range(self.epoch, self.epochs):
                if self.stop_training:
                    break
                self.on_epoch_begin()
                self.train_one_epoch(self.epoch)
                if self.validate_on_training_data:
                    self.train_val(self.epoch, "train_val")
                self.val(self.epoch, "validation")
                self.on_epoch_end()
                self.epoch += 1
            self.on_fit_end()
            self.val(self.epoch, "validation", evaluating_best_model=True)
            self.val(self.epoch, "test", evaluating_best_model=True)

        except KeyboardInterrupt:
            pprint("Training interrupted by user. Stopping training")
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.on_end()

    def train_one_epoch(self, epoch: int) -> None:
        """Train the model for one epoch.

        Args:
            epoch (int): The current epoch.
        """
        self.model.train()
        self.on_train_begin()
        num_batches = len(self.dataloaders["train"])
        for batch_idx, batch in enumerate(
            track(self.dataloaders["train"], description=f"Epoch: {epoch} | Training")
        ):
            with torch.autocast(
                device_type="cuda", enabled=self.use_amp, dtype=torch.bfloat16
            ):
                batch = batch.to(self.device)
                y_probs, targets, loss = self.loss_function(
                    batch,
                    model=self.model,
                    scale=self.gradient_scaler.get_scale(),
                    epoch=epoch,
                    explanation_decision_boundary=self.explanation_decision_boundary,
                    use_token_loss=self.config.loss.configs.get('use_token_loss', True),
                )
                loss = loss / self.accumulate_grad_batches
            self.gradient_scaler.scale(loss).backward()
            if ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (
                batch_idx + 1 == num_batches
            ):
                if self.config.trainer.clip_grad_norm:
                    self.gradient_scaler.unscale_(self.optimizer)
                    # torch.nn.utils.clip_grad_value_(norm=self.model.parameters(), clip_value=self.config.trainer.clip_value)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.trainer.clip_grad_norm
                    )
                self.gradient_scaler.step(self.optimizer)
                self.gradient_scaler.update()
                if self.lr_scheduler is not None:
                    if not isinstance(
                        self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.lr_scheduler.step()
                self.optimizer.zero_grad()
            self.update_metrics(
                y_probs=y_probs, targets=targets, loss=loss, split_name="train"
            )
        self.on_train_end(epoch)

    @torch.no_grad()
    def train_val(self, epoch, split_name: str = "train_val") -> None:
        """Validate on the training data. This is useful for testing for overfitting. Due to memory constraints, we donøt save the outputs.

        Args:
            epoch (_type_): _description_
            split_name (str, optional): _description_. Defaults to "train_val".
        """
        self.model.eval()
        self.on_val_begin()

        for batch in track(
            self.dataloaders[split_name],
            description=f"Epoch: {epoch} | Validating on training data",
        ):
            with torch.autocast(
                device_type="cuda", enabled=self.use_amp, dtype=torch.bfloat16
            ):
                y_probs, targets, loss = self.loss_function(
                    batch.to(self.device),
                    model=self.model,
                    explanation_decision_boundary=self.explanation_decision_boundary,
                )
            self.update_metrics(
                y_probs=y_probs, targets=targets, loss=loss, split_name=split_name
            )
        self.on_val_end(split_name, epoch)

    @torch.no_grad()
    def val(
        self, epoch, split_name: str = "validation", evaluating_best_model: bool = False
    ) -> None:
        self.model.eval()
        self.on_val_begin()
        y_probs_list = []
        targets_list = []
        y_probs_cpu = []
        targets_cpu = []
        ids = []

        for idx, batch in enumerate(
            track(
                self.dataloaders[split_name],
                description=f"Epoch: {epoch} | Validating on {split_name}",
            )
        ):
            with torch.autocast(
                device_type="cuda", enabled=self.use_amp, dtype=torch.bfloat16
            ):
                y_probs, targets, loss = self.loss_function(
                    batch.to(self.device),
                    model=self.model,
                    explanation_decision_boundary=self.explanation_decision_boundary,
                )
            self.update_metrics(
                y_probs=y_probs, targets=targets, loss=loss, split_name=split_name
            )
            y_probs_list.append(y_probs)
            targets_list.append(targets)
            ids.append(batch.ids)
            if idx % 1000 == 0:
                # move to cpu to save gpu memory
                y_probs_cpu.append(torch.cat(y_probs_list, dim=0).cpu())
                targets_cpu.append(torch.cat(targets_list, dim=0).cpu())
                y_probs_list = []
                targets_list = []
        y_probs_cpu.append(torch.cat(y_probs_list, dim=0).cpu())
        targets_cpu.append(torch.cat(targets_list, dim=0).cpu())

        y_probs = torch.cat(y_probs_cpu, dim=0)
        targets = torch.cat(targets_cpu, dim=0)
        ids = [item for sublist in ids for item in sublist]
        self.on_val_end(split_name, epoch, y_probs, targets, ids, evaluating_best_model)

    def update_metrics(
        self,
        y_probs: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        split_name: str,
    ) -> None:
        self.metric_collections[split_name].update(y_probs, targets, loss)

    def calculate_metrics(
        self,
        split_name: str,
        y_probs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        evaluating_best_model: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        results_dict: dict[str, dict[str, Any]] = defaultdict(dict)
        if split_name == "validation":
            results_dict[split_name] = self.metric_collections[split_name].compute()
        else:
            results_dict[split_name] = self.metric_collections[split_name].compute(
                y_probs, targets
            )

        if self.threshold_tuning and split_name == "validation":
            best_result, best_db = f1_score_db_tuning(y_probs, targets)
            results_dict[split_name] |= {"f1_micro_tuned": best_result}
            if evaluating_best_model:
                pprint(f"Best threshold: {best_db}")
                pprint(f"Best result: {best_result}")
                self.metric_collections["test"].set_threshold(best_db)
            self.best_db = best_db
        return results_dict

    def reset_metric(self, split_name: str) -> None:
        self.metric_collections[split_name].reset_metrics()

    def reset_metrics(self) -> None:
        for split_name in self.metric_collections.keys():
            self.metric_collections[split_name].reset_metrics()

    def on_initialisation_end(self) -> None:
        for callback in self.callbacks:
            callback.on_initialisation_end(self)

    def on_fit_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_begin(self)

    def on_fit_end(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_end(self)

    def on_train_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self, epoch: int) -> None:
        results_dict = self.calculate_metrics(split_name="train")
        results_dict["lr"] = self.optimizer.param_groups[0]["lr"]
        self.log_dict(results_dict, epoch)
        for callback in self.callbacks:
            callback.on_train_end()

    def on_val_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_val_begin()

    def on_val_end(
        self,
        split_name: str,
        epoch: int,
        logits: torch.Tensor = None,
        targets: torch.Tensor = None,
        ids: Optional[list[int]] = None,
        evaluating_best_model: bool = False,
    ) -> None:
        results_dict = self.calculate_metrics(
            split_name=split_name,
            y_probs=logits,
            targets=targets,
            evaluating_best_model=evaluating_best_model,
        )
        self.current_val_results = results_dict
        self.log_dict(results_dict, epoch)
        for callback in self.callbacks:
            callback.on_val_end()

        if evaluating_best_model:
            self.save_predictions(
                split_name=split_name, logits=logits, targets=targets, ids=ids
            )

    def save_predictions(
        self,
        split_name: str = "test",
        logits: torch.Tensor = None,
        targets: torch.Tensor = None,
        ids: Optional[list[int]] = None,
    ):
        target_tokenizer = self.lookups.target_tokenizer
        code_names = target_tokenizer.target_names()
        logits = logits.numpy()
        df = pd.DataFrame(logits, columns=code_names)
        df[TARGET_COLUMN] = list(map(target_tokenizer.torch_one_hot_decoder, targets))
        df[ID_COLUMN] = ids
        df.to_feather(self.experiment_path / f"predictions_{split_name}.feather")

    def on_epoch_begin(self) -> None:
        self.reset_metrics()
        for callback in self.callbacks:
            callback.on_epoch_begin(self)

    def on_epoch_end(self) -> None:
        if self.lr_scheduler is not None:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(
                    self.current_val_results["validation"]["f1_micro"]
                )

        for callback in self.callbacks:
            callback.on_epoch_end(self)

    def on_batch_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_end()

    def log_dict(
        self, nested_dict: dict[str, dict[str, torch.Tensor]], epoch: int
    ) -> None:
        if self.print_metrics:
            self.print(nested_dict)
        for callback in self.callbacks:
            callback.log_dict(nested_dict, epoch)

    def on_end(self) -> None:
        for callback in self.callbacks:
            callback.on_end()

    def print(self, nested_dict: dict[str, dict[str, Any]]) -> None:
        for split_name in nested_dict.keys():
            pprint(nested_dict[split_name])

    def to(self, device: str) -> "Trainer":
        self.model.to(device)
        for split_name in self.metric_collections.keys():
            self.metric_collections[split_name].to(device)

        if self.reference_model is not None:
            self.reference_model.to(device)
        self.device = device
        return self

    def save_checkpoint(self, file_name: str) -> None:
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.gradient_scaler.state_dict(),
            "epoch": self.epoch,
            "db": self.best_db,
            "explanation_decision_boundary": self.explanation_decision_boundary,
            "num_classes": self.lookups.data_info["num_classes"],
        }
        torch.save(checkpoint, self.experiment_path / file_name)
        pprint("Saved checkpoint to {}".format(self.experiment_path / file_name))

    def load_checkpoint(self, file_name: str) -> None:
        """Load a checkpoint from a file.
        
        Args:
            file_name (str): The name of the checkpoint file
        """
        checkpoint = torch.load(self.experiment_path / file_name)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.gradient_scaler.load_state_dict(checkpoint["scaler"])
        self.epoch = checkpoint["epoch"]
        self.best_db = checkpoint["db"]
        
        # Load explanation_decision_boundary if available
        if "explanation_decision_boundary" in checkpoint:
            self.explanation_decision_boundary = checkpoint["explanation_decision_boundary"]
            pprint(f"Loaded explanation decision boundary: {self.explanation_decision_boundary}")
        else:
            # Check if we need to calculate the explanation decision boundary
            is_masked_pooling = (self._original_loss_function.__name__ == 'masked_pooling_aux_loss' or 
                                self.config.loss.name == 'masked_pooling_aux_loss')
            
            need_threshold = False
            if is_masked_pooling:
                strategy = "auto"
                if hasattr(self.config.loss, 'configs') and hasattr(self.config.loss.configs, 'evidence_selection_strategy'):
                    strategy = self.config.loss.configs.evidence_selection_strategy
                
                if strategy != "evidence_ids":
                    need_threshold = True
                    
            if need_threshold:
                pprint("Explanation decision boundary not found in checkpoint. Calculating...")
                self.calculate_explanation_decision_boundary()
        
        pprint(f"Loaded checkpoint from {self.experiment_path / file_name}")

    def save_configs(self) -> None:
        self.lookups.target_tokenizer.save(
            self.experiment_path / "target_tokenizer.json"
        )
        OmegaConf.save(self.config, self.experiment_path / "config.yaml")
