from copy import deepcopy
from typing import Sequence

import torch
import sys

sys.path.append("./")
from explainable_medical_coding.explainability.helper_functions import (
    create_baseline_input,
)
from explainable_medical_coding.utils.datatypes import Batch

diet_gradient_scaler = torch.cuda.amp.GradScaler()
advesarial_noise_gradient_scaler = torch.cuda.amp.GradScaler()


def forward_embedding_noise(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Forward pass of the model when adding noise to the input token embeddings.

    Args:
        model (torch.nn.Module): Model to be used
        input_ids (torch.Tensor): Input token ids
        attention_mask (torch.Tensor): Attention mask
        noise (torch.Tensor): Noise to be added to the input token embeddings

    Returns:
        torch.Tensor: Logits
    """
    chunked_embeddings = model.get_chunked_embedding(input_ids)
    logits = model.forward_embedding_input(
        chunked_embeddings + noise, attention_masks=attention_mask
    )
    return logits


def kl_divergence(
    attention: torch.Tensor, one_hot_evidence: torch.Tensor
) -> torch.Tensor:
    """Calculate the KL divergence loss between the attention and the one hot encoded evidence

    Args:
        attention (torch.Tensor): Attention tensor [number_of_targets, sequence_length]
        one_hot_evidence (torch.Tensor): One hot encoded evidence tensor [number_of_targets, sequence_length]

    Returns:
        torch.Tensor: KL divergence lossPatie
    """
    # sum along the sequence length
    attention = torch.log_softmax(attention, dim=1)  # [num_classes, seq_len]
    one_hot_evidence = torch.nn.functional.normalize(one_hot_evidence, p=1.0, dim=-1)
    kl = torch.nn.functional.kl_div(attention, one_hot_evidence, reduction="mean")
    if torch.isnan(kl):
        raise ValueError("KL loss is NaN")

    return kl


def one_hot_encode_evidence_token_ids(
    evidence_token_ids: Sequence[Sequence[int]], max_length: int
) -> torch.Tensor:
    output_tensor = torch.zeros((len(evidence_token_ids), max_length))
    for target_idx, target_evidence_token_ids in enumerate(evidence_token_ids):
        if len(target_evidence_token_ids) > 0:
            output_tensor[target_idx, target_evidence_token_ids] = 1
    return output_tensor.float()


def binary_cross_entropy_loss(
    batch: Batch,
    model: torch.nn.Module,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, targets, attention_masks = (
        batch.input_ids,
        batch.targets,
        batch.attention_masks,
    )
    logits = model(input_ids, attention_masks)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return torch.sigmoid(logits), targets, loss


def kl_attention_loss(
    batch: Batch,
    model: torch.nn.Module,
    lambda_1: float = 2.5,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, targets, attention_masks, evidence_token_ids = (
        batch.input_ids,
        batch.targets,
        batch.attention_masks,
        batch.evidence_input_ids,
    )
    logits, label_wise_attentions = model(
        input_ids, attention_masks, output_attentions=True
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    if torch.isnan(loss):
        raise ValueError("Loss is NaN")

    if evidence_token_ids is None:
        return torch.sigmoid(logits), targets, loss

    sequence_length = label_wise_attentions.shape[2]
    batch_size = label_wise_attentions.shape[0]

    kl_loss = 0
    num_elements_with_evidence = 0
    for element_idx in range(batch_size):
        # If element has no evidence tokens, skip it
        if evidence_token_ids[element_idx] is None:
            continue

        target_ids = torch.where(targets[element_idx])[0]
        one_hot_evidence = one_hot_encode_evidence_token_ids(
            evidence_token_ids[element_idx], sequence_length
        ).to(input_ids.device)
        kl_loss += kl_divergence(
            label_wise_attentions[element_idx, target_ids], one_hot_evidence
        )
        num_elements_with_evidence += 1

    if num_elements_with_evidence > 0:
        loss += lambda_1 * kl_loss / (num_elements_with_evidence)
    return torch.sigmoid(logits), targets, loss


def double_backpropagation_loss(
    batch: Batch,
    model: torch.nn.Module,
    lambda_1: float = 1.0,
    p: int = 1,
    scale: int = 1024,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = 1024
    input_ids, targets, attention_masks = (
        batch.input_ids,
        batch.targets,
        batch.attention_masks,
    )
    logits = model(input_ids, attention_masks)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    if torch.isnan(loss):
        import pdb

        pdb.set_trace()
        raise ValueError("Loss is NaN")

    if loss.requires_grad:
        scaled_loss = loss * scale
        input_grad = torch.autograd.grad(
            scaled_loss, model.parameters(), create_graph=True
        )[0]
        input_grad = input_grad / scale
        loss_dbp = torch.norm(input_grad, p=2, dim=1).sum()
        if torch.isnan(loss_dbp):
            print("Double back propagation loss is NaN. Skipping regularization.")
        else:
            loss += lambda_1 * loss_dbp

    return torch.sigmoid(logits), targets, loss


def advesarial_training_loss(
    batch: Batch,
    model: torch.nn.Module,
    epsilon=1e-5,
    alpha=1,
    num_iter=5,
    lambda_1: float = 0.01,
    lambda_2: float = 0.5,
    scale: int = 1024,
    attack_type: str = "pgd",
    norm: str = "softmax",
    adv_dist: str = "kl",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, targets, attention_masks = (
        batch.input_ids,
        batch.targets,
        batch.attention_masks,
    )
    advesarial_model = deepcopy(model)
    advesarial_model.zero_grad()
    advesarial_model.eval()
    if attack_type == "fgsm":
        advesarial_noise = fgsm(
            advesarial_model,
            input_ids,
            attention_masks,
            targets,
            epsilon=epsilon,
            scale=scale,
        )
    elif attack_type == "pgd":
        advesarial_noise = pgd(
            advesarial_model,
            input_ids,
            attention_masks,
            targets,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            scale=scale,
        )
    elif attack_type == "pgd_adam":
        advesarial_noise = pgd_adam(
            advesarial_model,
            input_ids,
            attention_masks,
            targets,
            epsilon=epsilon,
            alpha=alpha,
            scale=scale,
        )
        scale = advesarial_noise_gradient_scaler.get_scale()

    logits = model(input_ids, attention_masks)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    if torch.isnan(loss):
        import pdb

        pdb.set_trace()
        raise ValueError("Loss is NaN")

    logits_adv = forward_embedding_noise(
        model, input_ids, attention_masks, advesarial_noise
    )

    # symetric kl divergence
    if adv_dist == "kl":
        if norm == "softmax":
            advesarial_loss = torch.nn.functional.kl_div(
                torch.log_softmax(logits_adv, dim=-1, dtype=torch.float32),
                torch.softmax(logits.detach(), dim=-1, dtype=torch.float32),
                reduction="batchmean",
            ) + torch.nn.functional.kl_div(
                torch.log_softmax(logits, dim=-1, dtype=torch.float32),
                torch.softmax(logits_adv.detach(), dim=-1, dtype=torch.float32),
                reduction="batchmean",
            )
        elif norm == "l1":
            logits_adv_norm = torch.nn.functional.normalize(logits_adv, p=1.0, dim=-1)
            logits_norm = torch.nn.functional.normalize(logits, p=1.0, dim=-1)
            advesarial_loss = torch.nn.functional.kl_div(
                torch.log(logits_adv_norm), logits_norm.detach(), reduction="batchmean"
            ) + torch.nn.functional.kl_div(
                torch.log(logits_norm), logits_adv_norm.detach(), reduction="batchmean"
            )
        else:
            raise ValueError(f"Norm {norm} not supported")

    else:
        advesarial_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_adv, targets
        )

    # input gradient regularization

    if loss.requires_grad and (lambda_1 > 0):
        scaled_loss = loss * scale
        input_grad = torch.autograd.grad(
            scaled_loss, model.parameters(), create_graph=True
        )[0]
        input_grad = input_grad / scale
        loss_dbp = torch.norm(input_grad, p=2, dim=1).sum()

        if torch.isnan(loss_dbp):
            print("Double back propagation loss is NaN. Skipping regularization.")
        else:
            loss += lambda_1 * loss_dbp

    if torch.isnan(advesarial_loss):
        print("Advesarial loss is NaN. Skipping regularization.")
    else:
        loss += lambda_2 * advesarial_loss

    return torch.sigmoid(logits), targets, loss


def masking_loss(
    batch: Batch,
    model: torch.nn.Module,
    epoch: int = 0,
    total_epochs: int = 5,
    lambda_1: float = 1,
    diet_constant: float = 1,
    baseline_token_id: int = 50001,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    pad_token_id: int = 1,
    lr: float = 0.1,
    rounding_schedule: bool = True,
    distillation: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    advesarial_model = deepcopy(model)
    advesarial_model.zero_grad()
    advesarial_model.eval()
    input_ids, targets, attention_masks, teacher_logits = (
        batch.input_ids,
        batch.targets,
        batch.attention_masks,
        batch.teacher_logits,
    )
    baseline = create_baseline_input(
        input_ids,
        baseline_token_id=baseline_token_id,
        cls_token_id=baseline_token_id,
        eos_token_id=baseline_token_id,
    )
    baseline = baseline.requires_grad_(False)
    baseline_embeddings = model.get_chunked_embedding(baseline).detach()
    mask = token_masking(
        advesarial_model,
        input_ids,
        attention_masks,
        baseline_embeddings,
        diet_constant=diet_constant,
        lr=lr,
    )
    mask = mask.requires_grad_(False)
    mask = mask.view(input_ids.size(0), -1, 1)[:, : input_ids.size(1)].squeeze(-1)

    if rounding_schedule:
        rounding_constant = max(0.4 - epoch * (0.4 / total_epochs), 0)
    else:
        rounding_constant = 0

    mask = torch.round(mask + rounding_constant)
    masked_input_ids = (input_ids * mask + baseline_token_id * (1 - mask)).to(
        torch.long
    )
    masked_input_ids[:, 0] = cls_token_id
    masked_input_ids[torch.where(input_ids == eos_token_id)] = eos_token_id
    masked_input_ids[torch.where(input_ids == pad_token_id)] = pad_token_id

    student_logits = model(input_ids, attention_masks)
    student_probs = torch.sigmoid(student_logits)

    teacher_probs = torch.sigmoid(teacher_logits)

    masked_student_logits = model(masked_input_ids, attention_masks)
    masked_student_probs = torch.sigmoid(masked_student_logits)

    batch_size = input_ids.size(0)
    if distillation:
        model_loss = (
            torch.norm(student_probs - teacher_probs.detach(), p=1) / batch_size
        )
        masked_loss = (
            torch.norm(student_probs.detach() - masked_student_probs, p=1) / batch_size
        )
    else:
        model_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            student_logits, targets
        )
        masked_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            masked_student_logits, targets
        )

    if torch.isnan(masked_loss):
        print("Diet loss is NaN. Skipping regularization.")
        loss = model_loss
    else:
        loss = model_loss + lambda_1 * masked_loss

    if torch.isnan(loss):
        import pdb

        pdb.set_trace()
        raise ValueError("Loss is NaN")

    return student_probs, targets, loss


@torch.enable_grad()
def pgd(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_iter: int,
    scale: int = 1024,
):
    """Construct FGSM adversarial examples on the examples X"""
    chunked_embeddings = model.get_chunked_embedding(input_ids).detach()

    # delta = torch.zeros_like(chunked_embeddings, requires_grad=True)
    delta = torch.rand_like(chunked_embeddings, requires_grad=True)
    delta.data = delta.data * 2 * epsilon - epsilon

    logits = model.forward_embedding_input(
        chunked_embeddings, attention_masks=attention_mask
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    for t in range(num_iter):
        logits = model.forward_embedding_input(
            chunked_embeddings + torch.tanh(delta) * epsilon,
            attention_masks=attention_mask,
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        print(loss)
        loss = loss * scale
        loss.backward()
        delta.data = delta + chunked_embeddings.shape[0] * alpha * delta.grad.data / (
            scale * delta.grad.data.max(dim=-1, keepdim=True).values
        )
        model.zero_grad()
        delta.grad.zero_()
    return torch.tanh(delta.detach()) * epsilon


@torch.enable_grad()
def token_masking(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    baseline_embeddings: torch.Tensor,
    diet_constant: float = 0.1,
    lr: float = 0.1,
    max_iter: int = 20,
):
    """Construct FGSM adversarial examples on the examples X"""
    prev_loss, prev_prev_loss = float("inf"), float("inf")
    mask_converged = False
    is_training = model.training

    if is_training:
        model.eval()

    chunked_embeddings = model.get_chunked_embedding(input_ids).detach()

    original_logits = model.forward_embedding_input(
        chunked_embeddings, attention_masks=attention_mask
    )

    original_y_probs = torch.sigmoid(original_logits)
    # delta = torch.zeros_like(chunked_embeddings, requires_grad=True)
    mask = torch.ones(
        (chunked_embeddings.size(0), chunked_embeddings.size(1), 1),
        requires_grad=True,
        device=chunked_embeddings.device,
    )
    optimizer = torch.optim.AdamW([mask], lr=lr)
    mask.data = mask.data * 2

    for idx in range(max_iter):
        if mask_converged:
            break
        sigmoid_mask = torch.sigmoid(mask)
        input_embeddings = chunked_embeddings * sigmoid_mask + baseline_embeddings * (
            1 - sigmoid_mask
        )
        logits = model.forward_embedding_input(
            input_embeddings, attention_masks=attention_mask
        )

        y_probs = torch.sigmoid(logits)
        loss = torch.norm(sigmoid_mask, p=1) / (
            mask.size(0) * mask.size(1)
        ) + diet_constant * torch.norm(
            y_probs - original_y_probs.detach(), p=1
        ) / logits.size(0)
        mask_converged = (loss > 0.97 * prev_prev_loss) and (
            loss < 1.03 * prev_prev_loss
        )

        prev_prev_loss = prev_loss
        prev_loss = loss

        loss = diet_gradient_scaler.scale(loss)
        loss.backward()
        diet_gradient_scaler.step(optimizer)
        diet_gradient_scaler.update()

        optimizer.zero_grad()
        model.zero_grad()

    if is_training:
        model.train()

    return torch.sigmoid(mask.detach())


@torch.enable_grad()
def pgd_attack(
    input_ids,
    attention_mask,
    targets,
    model,
    epsilon=0.01,
    alpha=1,
    num_iter=5,
    scale=1024,
):
    noise = pgd(
        model,
        input_ids,
        attention_mask,
        targets,
        epsilon=epsilon,
        alpha=alpha,
        num_iter=num_iter,
        scale=scale,
    )
    chunked_embeddings = model.get_chunked_embedding(input_ids).detach()
    logits = model.forward_embedding_input(
        chunked_embeddings + noise, attention_masks=attention_mask
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return torch.sigmoid(logits).detach(), loss.detach()


@torch.enable_grad()
def pgd_adam(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    alpha: float,
    scale: int = 1024,
):
    is_training = model.training
    if is_training:
        model.eval()
    """Construct FGSM adversarial examples on the examples X"""
    chunked_embeddings = model.get_chunked_embedding(input_ids).detach()

    # delta = torch.zeros_like(chunked_embeddings, requires_grad=True)
    delta = torch.rand_like(chunked_embeddings, requires_grad=True)
    delta.data = delta.data * 2 * epsilon - epsilon
    optimizer = torch.optim.AdamW([delta], lr=alpha)
    prev_loss, prev_prev_loss = float("inf"), float("inf")
    mask_converged = False
    for _ in range(100):
        if mask_converged:
            break
        optimizer.zero_grad()
        logits = model.forward_embedding_input(
            chunked_embeddings + torch.tanh(delta) * epsilon,
            attention_masks=attention_mask,
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        mask_converged = (loss > 0.995 * prev_prev_loss) and (
            loss < 1.005 * prev_prev_loss
        )
        prev_prev_loss = prev_loss
        prev_loss = loss

        loss = advesarial_noise_gradient_scaler.scale(-loss)
        loss.backward()
        advesarial_noise_gradient_scaler.step(optimizer)
        advesarial_noise_gradient_scaler.update()
        model.zero_grad()
    if is_training:
        model.train()
    return torch.tanh(delta.detach()) * epsilon


@torch.enable_grad()
def fgsm(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    scale: int = 1024,
):
    """Construct FGSM adversarial examples on the examples X"""
    chunked_embeddings = model.get_chunked_embedding(input_ids).detach()

    delta = torch.zeros_like(chunked_embeddings, requires_grad=True)
    # delta = torch.rand_like(chunked_embeddings, requires_grad=True)
    # delta.data = delta.data * 2 * epsilon - epsilon
    logits = model.forward_embedding_input(
        chunked_embeddings + delta, attention_masks=attention_mask
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss = loss * scale
    loss.backward()
    delta = epsilon * delta.grad.detach().sign()
    logits = model.forward_embedding_input(
        chunked_embeddings + delta, attention_masks=attention_mask
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    return delta

def _build_effective_attention_mask_from_batch(batch, seq_len: int, device: torch.device):
    """Return a (B, NC, L) 0/1 mask for class-specific attention if available, else None."""
    if hasattr(batch, "effective_attention_mask") and batch.effective_attention_mask is not None:
        effective_masks = batch.effective_attention_mask
        
        if isinstance(effective_masks, torch.Tensor):
            # Already a tensor (should be the case after collate_fn processing)
            if effective_masks.dim() == 3:  # (B, NC, L)
                # Pad sequence dimension if needed
                if effective_masks.size(2) < seq_len:
                    pad_len = seq_len - effective_masks.size(2)
                    effective_masks = torch.nn.functional.pad(
                        effective_masks, (0, pad_len), value=0.0
                    )
                elif effective_masks.size(2) > seq_len:
                    effective_masks = effective_masks[:, :, :seq_len]
                
                return effective_masks.to(device)
        elif isinstance(effective_masks, list):
            # Fallback: Convert list of (NC, L) tensors to (B, NC, L) tensor
            if len(effective_masks) > 0 and effective_masks[0] is not None:
                # Stack all masks along batch dimension
                effective_mask_tensor = torch.stack(effective_masks, dim=0)  # (B, NC, L)
                
                # Pad sequence dimension if needed
                if effective_mask_tensor.size(2) < seq_len:
                    pad_len = seq_len - effective_mask_tensor.size(2)
                    effective_mask_tensor = torch.nn.functional.pad(
                        effective_mask_tensor, (0, pad_len), value=0.0
                    )
                elif effective_mask_tensor.size(2) > seq_len:
                    effective_mask_tensor = effective_mask_tensor[:, :, :seq_len]
                
                return effective_mask_tensor.to(device)
    
    return None

def _build_selected_mask_from_batch(batch, seq_len: int, device: torch.device):
    """Return a (B, L) 0/1 mask if available, else None.
    Tries batch.selected_token_mask first; otherwise unions indices in batch.evidence_input_ids.
    """

    """Return a (B, L) 0/1 mask if available, else None."""
    # 1) Direct mask if available
    if hasattr(batch, "selected_mask_ids") and batch.selected_mask_ids is not None:
        selected_mask_ids = batch.selected_mask_ids
        
        # Handle both tensor and list cases
        if isinstance(selected_mask_ids, list):
            # Convert list of lists to tensor mask
            B = len(selected_mask_ids)
            mask = torch.zeros((B, seq_len), device=device, dtype=torch.float32)
            
            for b, token_indices in enumerate(selected_mask_ids):
                for i in token_indices:
                    if 0 <= i < seq_len:
                        mask[b, i] = 1.0
            return mask
        else:
            # Already a tensor - convert from indices to binary mask
            if isinstance(selected_mask_ids, torch.Tensor):
                expected_B = batch.input_ids.size(0)
                mask = torch.zeros((expected_B, seq_len), device=device, dtype=torch.float32)
                
                for b in range(expected_B):
                    for i in selected_mask_ids[b]:
                        idx = int(i.item())  # Convert tensor element to int
                        if 0 <= idx < seq_len:
                            mask[b, idx] = 1.0
                return mask
            else:
                print("Warning: selected_mask_ids is neither a list nor a tensor. Falling back to other strategies.")
                return None
    else:
        print("Warning: selected_mask_ids not in batch or is None. Falling back to other strategies.")

    # 2) Build from evidence_input_ids (list per sample -> list per label -> token idx list)
    evid = getattr(batch, "evidence_input_ids", None)
    if evid is not None:
        B = len(evid)
        mask = torch.zeros((B, seq_len), device=device, dtype=torch.float32)
        for b in range(B):
            lists_per_label = evid[b] or []
            for token_idx_list in lists_per_label or []:
                if token_idx_list:
                    for i in token_idx_list:
                        if 0 <= i < seq_len:
                            mask[b, i] = 1.0
        return mask
    return None

def masked_pooling_aux_loss(
    batch,
    model,
    lambda_aux: float = 0.3,
    stop_gradient_unselected: bool = True,
    mask_pooling: bool = True,                       # NEW
    soft_alpha: float = 0.0,                         # NEW
    fallback_to_full_attention_if_empty: bool = True,# NEW
    **kwargs,
):
    """Document-level BCE + token-level auxiliary BCE on selected tokens.

    - Masked pooling: only selected tokens contribute to label-wise pooling (or soft-weighted).
    - Optional stop-grad: block gradients through unselected token reps.
    - Auxiliary: token head predicts labels at each token; BCE over selected positions only.
    - If no selected tokens for a sample and fallback=True, we keep original attention for that sample.

    Returns: (y_probs, targets, total_loss)
    """
    input_ids, targets, attention_masks = batch.input_ids, batch.targets, batch.attention_masks

    seq_len = input_ids.size(1)
    sel_mask = _build_selected_mask_from_batch(batch, seq_len, input_ids.device)
    effective_mask = _build_effective_attention_mask_from_batch(batch, seq_len, input_ids.device)

    # Use effective mask if available, otherwise fall back to regular attention masks

    masks_to_use = effective_mask if effective_mask is not None else attention_masks

    # Forward with masked pooling; also request token-level logits for the aux loss
    doc_logits, tok_logits = model.forward_with_selected_tokens(
        input_ids=input_ids,
        attention_masks=attention_masks,
        selected_token_mask=sel_mask,
        stop_gradient_unselected=stop_gradient_unselected,
        return_token_logits=lambda_aux>0,
        output_attentions=False,
        mask_pooling=mask_pooling,
        soft_alpha=soft_alpha,
        fallback_to_full_attention_if_empty=fallback_to_full_attention_if_empty,
        effective_attention_mask=masks_to_use,
    )

    # Primary document-level loss
    doc_loss = torch.nn.functional.binary_cross_entropy_with_logits(doc_logits, targets)
    total_loss = doc_loss

    # Auxiliary token-level loss (only if we have a selection mask and token logits)
    if (sel_mask is not None) and (tok_logits is not None):
        B, L, C = tok_logits.shape
        tgt_broadcast = targets.unsqueeze(1).expand(B, L, C)  # (B, L, C)

        aux_loss_all = torch.nn.functional.binary_cross_entropy_with_logits(
            tok_logits, tgt_broadcast, reduction="none"
        )  # (B, L, C)
        sel_bool = sel_mask.to(dtype=torch.bool)
        pad_len = aux_loss_all.size(1) - sel_bool.size(1)
        if pad_len > 0:
            sel_bool = torch.nn.functional.pad(sel_bool, (0, pad_len), value=False)
        if sel_bool.any():
            sel_3d = sel_bool.unsqueeze(-1).expand_as(aux_loss_all)  # (B, L, C)
            aux_loss = aux_loss_all[sel_3d].mean()
            total_loss = total_loss + lambda_aux * aux_loss

    return torch.sigmoid(doc_logits), targets, total_loss
