import gc
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable, Optional

import pandas as pd
import torch
from accelerate import Accelerator
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


class Trainer2GPU:
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
        # Initialize Accelerator for multi-GPU training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=accumulate_grad_batches,
            mixed_precision="bf16" if config.trainer.use_amp else "no",
            log_with=None,  # Can add wandb, tensorboard etc. if needed
        )
        
        self.config = config
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.callbacks = callbacks
        self.metric_collections = metric_collections
        self.lr_scheduler = lr_scheduler
        self.lookups = lookups
        self.accumulate_grad_batches = accumulate_grad_batches
        
        # Prepare model, optimizer, and dataloaders with Accelerate
        (
            self.model,
            self.optimizer,
            self.dataloaders["train"],
            self.dataloaders["validation"],
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.dataloaders["train"],
            self.dataloaders["validation"],
        )
        
        # Prepare test dataloader if it exists
        if "test" in self.dataloaders:
            self.dataloaders["test"] = self.accelerator.prepare(self.dataloaders["test"])
            
        # Prepare train_val dataloader if it exists
        if "train_val" in self.dataloaders:
            self.dataloaders["train_val"] = self.accelerator.prepare(self.dataloaders["train_val"])
        
        # Prepare lr_scheduler if it exists
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        
        # Move metric collections to accelerator device
        for split_name in self.metric_collections.keys():
            self.metric_collections[split_name].to(self.accelerator.device)
        
        if self.accelerator.is_main_process:
            pprint(f"Using {self.accelerator.num_processes} GPUs for training")
            pprint(f"Accumulating gradients over {self.accumulate_grad_batches} batch(es).")
        
        self.validate_on_training_data = config.trainer.validate_on_training_data
        self.print_metrics = config.trainer.print_metrics
        self.epochs = config.trainer.epochs
        self.epoch = 0
        self.use_amp = config.trainer.use_amp
        self.threshold_tuning = config.trainer.threshold_tuning
        self.experiment_path = Path(mkdtemp())
        self.current_val_results: dict[str, dict[str, torch.Tensor]] = {}
        self.stop_training = False
        self.best_db = 0.5
        self.on_initialisation_end()

    def fit(self) -> None:
        """Train and validate the model."""
        try:
            if self.accelerator.is_main_process:
                self.save_configs()
            self.on_fit_begin()
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
            if self.accelerator.is_main_process:
                pprint("Training interrupted by user. Stopping training")
        
        # Clean up
        self.accelerator.wait_for_everyone()
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.on_end()

    def train_one_epoch(self, epoch: int) -> None:
        """Train the model for one epoch.

        Args:
            epoch (int): The current epoch.
        """
        self.model.train()
        self.on_train_begin()
        
        # Use accelerator's progress bar only on main process
        dataloader = self.dataloaders["train"]
        if self.accelerator.is_main_process:
            dataloader = track(dataloader, description=f"Epoch: {epoch} | Training")
        
        for batch_idx, batch in enumerate(dataloader):
            with self.accelerator.accumulate(self.model):
                # No need for manual device placement - Accelerate handles it
                y_probs, targets, loss = self.loss_function(
                    batch,
                    model=self.model,
                    scale=1.0,  # Accelerate handles scaling
                    epoch=epoch,
                )
                
                # Accelerate handles gradient accumulation and scaling
                self.accelerator.backward(loss)
                
                if self.config.trainer.clip_grad_norm:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.config.trainer.clip_grad_norm
                    )
                
                self.optimizer.step()
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
        """Validate on the training data. This is useful for testing for overfitting.

        Args:
            epoch: Current epoch
            split_name (str, optional): Split name. Defaults to "train_val".
        """
        self.model.eval()
        self.on_val_begin()

        dataloader = self.dataloaders[split_name]
        if self.accelerator.is_main_process:
            dataloader = track(
                dataloader,
                description=f"Epoch: {epoch} | Validating on training data",
            )

        for batch in dataloader:
            y_probs, targets, loss = self.loss_function(batch, model=self.model)
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
        ids_list = []

        dataloader = self.dataloaders[split_name]
        if self.accelerator.is_main_process:
            dataloader = track(
                dataloader,
                description=f"Epoch: {epoch} | Validating on {split_name}",
            )

        for batch in dataloader:
            y_probs, targets, loss = self.loss_function(batch, model=self.model)
            
            # Gather predictions from all processes
            y_probs_gathered = self.accelerator.gather_for_metrics(y_probs)
            targets_gathered = self.accelerator.gather_for_metrics(targets)
            ids_gathered = self.accelerator.gather_for_metrics(batch.ids)
            
            self.update_metrics(
                y_probs=y_probs, targets=targets, loss=loss, split_name=split_name
            )
            
            # Only collect on main process to avoid memory issues
            if self.accelerator.is_main_process:
                y_probs_list.append(y_probs_gathered.cpu())
                targets_list.append(targets_gathered.cpu())
                ids_list.extend(ids_gathered.cpu().tolist())

        # Concatenate results only on main process
        if self.accelerator.is_main_process:
            y_probs = torch.cat(y_probs_list, dim=0) if y_probs_list else None
            targets = torch.cat(targets_list, dim=0) if targets_list else None
            ids = ids_list
        else:
            y_probs = None
            targets = None
            ids = None

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
        
        # Only calculate metrics on main process to avoid issues with distributed metrics
        if self.accelerator.is_main_process:
            if split_name == "validation":
                results_dict[split_name] = self.metric_collections[split_name].compute()
            else:
                results_dict[split_name] = self.metric_collections[split_name].compute(
                    y_probs, targets
                )

            if self.threshold_tuning and split_name == "validation" and y_probs is not None:
                best_result, best_db = f1_score_db_tuning(y_probs, targets)
                results_dict[split_name] |= {"f1_micro_tuned": best_result}
                if evaluating_best_model:
                    pprint(f"Best threshold: {best_db}")
                    pprint(f"Best result: {best_result}")
                    self.metric_collections["test"].set_threshold(best_db)
                self.best_db = best_db
        
        # Broadcast best_db to all processes
        if self.threshold_tuning and split_name == "validation":
            if self.accelerator.is_main_process:
                best_db_tensor = torch.tensor(self.best_db, device=self.accelerator.device)
            else:
                best_db_tensor = torch.zeros(1, device=self.accelerator.device)
            
            # Broadcast from main process to all processes
            self.accelerator.broadcast(best_db_tensor, from_process=0)
            self.best_db = best_db_tensor.item()
        
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
        if self.accelerator.is_main_process and results_dict:
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
        
        if self.accelerator.is_main_process and results_dict:
            self.log_dict(results_dict, epoch)
        
        for callback in self.callbacks:
            callback.on_val_end()

        if evaluating_best_model and self.accelerator.is_main_process:
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
        if not self.accelerator.is_main_process:
            return
            
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
                # Need to ensure all processes have the same metric value
                if self.accelerator.is_main_process and self.current_val_results:
                    metric_value = self.current_val_results["validation"]["f1_micro"]
                else:
                    metric_value = torch.tensor(0.0, device=self.accelerator.device)
                
                # Broadcast metric to all processes
                metric_tensor = torch.tensor(metric_value, device=self.accelerator.device)
                self.accelerator.broadcast(metric_tensor, from_process=0)
                self.lr_scheduler.step(metric_tensor.item())

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
        if self.print_metrics and self.accelerator.is_main_process:
            self.print(nested_dict)
        for callback in self.callbacks:
            callback.log_dict(nested_dict, epoch)

    def on_end(self) -> None:
        for callback in self.callbacks:
            callback.on_end()

    def print(self, nested_dict: dict[str, dict[str, Any]]) -> None:
        for split_name in nested_dict.keys():
            pprint(nested_dict[split_name])

    def to(self, device: str) -> "Trainer2GPU":
        # With Accelerate, device management is handled automatically
        # This method is kept for compatibility but doesn't need to do anything
        if self.accelerator.is_main_process:
            pprint(f"Device management is handled by Accelerate. Using {self.accelerator.device}")
        return self

    def save_checkpoint(self, file_name: str) -> None:
        if not self.accelerator.is_main_process:
            return
            
        # Use Accelerate's save method to handle distributed state
        self.accelerator.wait_for_everyone()
        checkpoint = {
            "model": self.accelerator.get_state_dict(self.model),
            "epoch": self.epoch,
            "db": self.best_db,
            "num_classes": self.lookups.data_info["num_classes"],
        }
        
        # Save optimizer and lr_scheduler state
        checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
            
        torch.save(checkpoint, self.experiment_path / file_name)
        pprint("Saved checkpoint to {}".format(self.experiment_path / file_name))

    def load_checkpoint(self, file_name: str) -> None:
        checkpoint = torch.load(self.experiment_path / file_name, map_location=self.accelerator.device)
        
        # Load model state
        self.accelerator.load_state(checkpoint["model"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Load lr_scheduler state if it exists
        if self.lr_scheduler is not None and "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            
        self.epoch = checkpoint["epoch"]
        self.best_db = checkpoint["db"]
        
        if self.accelerator.is_main_process:
            pprint("Loaded checkpoint from {}".format(self.experiment_path / file_name))

    def save_configs(self) -> None:
        if not self.accelerator.is_main_process:
            return
            
        self.lookups.target_tokenizer.save(
            self.experiment_path / "target_tokenizer.json"
        )
        OmegaConf.save(self.config, self.experiment_path / "config.yaml")
