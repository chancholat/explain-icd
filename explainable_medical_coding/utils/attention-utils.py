# ---------------- Attention-based token selection functions ----------------
@torch.no_grad()
def find_optimal_attention_threshold(
    label_att: torch.Tensor,             # (B, C, L) label-wise attention weights
    attention_mask: torch.Tensor,        # (B, L) 0/1 padding mask
    evidence_masks: torch.Tensor,        # (B, L) 1 for evidence tokens, 0 otherwise
    label_probs: Optional[torch.Tensor] = None,  # (B, C) label probabilities
    strategy: str = "max",
    probability_threshold: float = 0.5,
    num_thresholds: int = 101,
) -> float:
    """
    Find the optimal attention threshold that maximizes F1 score between attention-based
    selection and evidence tokens.
    
    Args:
        label_att: Label-wise attention weights (B, C, L)
        attention_mask: Padding mask (B, L)
        evidence_masks: Ground truth evidence tokens (B, L)
        label_probs: Label probabilities (B, C) if using predicted labels
        strategy: Strategy for combining attention across labels ("max", "pred_weighted", "predicted_labels")
        probability_threshold: Threshold for label probabilities when using "predicted_labels" strategy
        num_thresholds: Number of threshold values to try
        
    Returns:
        The optimal attention threshold value
    """
    att = label_att.detach()
    # Normalize if needed
    if att.dim() != 3:
        raise ValueError(f"label_att must be (B, C, L), got {tuple(att.shape)}")
    sums = att.float().sum(dim=2, keepdim=True) + 1e-12
    if not torch.allclose(sums.mean(), torch.ones(1, device=att.device), atol=1e-1):
        att = torch.softmax(att, dim=2)
    
    # Convert attention to token scores based on strategy
    if strategy == "predicted_labels" and label_probs is not None:
        batch_size, num_classes, seq_len = att.shape
        # Get predicted labels (probability >= threshold)
        pred_mask = (label_probs >= probability_threshold).float()  # (B, C)
        
        # For each batch item, compute max attention across predicted labels
        scores = torch.zeros((batch_size, seq_len), device=att.device)
        for b in range(batch_size):
            pred_labels = torch.where(pred_mask[b] > 0)[0]
            if len(pred_labels) > 0:
                scores[b], _ = att[b, pred_labels].max(dim=0)
            else:
                scores[b], _ = att[b].max(dim=0)
                
    elif strategy == "pred_weighted" and label_probs is not None:
        # Weight attention by label probabilities
        p = label_probs / (label_probs.sum(dim=1, keepdim=True) + 1e-8)  # (B, C)
        scores = (p.unsqueeze(-1) * att).sum(dim=1)  # (B, L)
    else:
        # Default: max attention across labels
        scores, _ = att.max(dim=1)  # (B, L)
    
    # Apply attention mask
    scores = scores.clamp(0, 1) * attention_mask.to(scores.dtype)
    
    # Try different thresholds to find optimal F1
    thresholds = torch.linspace(0.01, 0.99, num_thresholds, device=scores.device)
    f1_scores = []
    
    for threshold in thresholds:
        # Create binary mask with current threshold
        sel = (scores >= threshold) & attention_mask.bool()
        
        # Calculate TP, FP, FN
        tp = (sel & evidence_masks.bool()).sum().float()
        fp = (sel & ~evidence_masks.bool()).sum().float()
        fn = (~sel & evidence_masks.bool()).sum().float()
        
        # Calculate F1
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_scores.append(f1.item())
    
    # Return threshold with best F1
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx].item()


@torch.no_grad()
def build_attention_selection_mask(
    model: torch.nn.Module,
    input_ids: torch.Tensor,              # (B, L)
    attention_mask: torch.Tensor,         # (B, L)
    strategy: str = "max",                # "max", "pred_weighted", or "predicted_labels" 
    threshold: Optional[float] = None,    # Fixed threshold or None to use adaptive threshold
    fallback_top1: bool = True,           # Select top token if no tokens exceed threshold
    probability_threshold: float = 0.5,   # For "predicted_labels" strategy
    evidence_masks: Optional[torch.Tensor] = None,  # (B, L) for threshold tuning
) -> torch.Tensor:
    """
    Build a token selection mask based on model's attention weights.
    
    Args:
        model: Model with attention mechanism
        input_ids: Input token ids
        attention_mask: Attention mask for padding
        strategy: Strategy for combining attention across labels
        threshold: Attention threshold (if None and evidence_masks provided, will tune automatically)
        fallback_top1: Whether to select top token if no tokens exceed threshold
        probability_threshold: Threshold for predicted labels strategy
        evidence_masks: Optional ground truth evidence masks for threshold tuning
        
    Returns:
        Binary selection mask (B, L) with 1 for selected tokens
    """
    # Get attention weights and logits
    model.eval()
    with torch.no_grad():
        try:
            # Try to get encoder attention directly
            _, _, label_att = model.get_encoder_attention_and_hidden_states(
                input_ids=input_ids, attention_masks=attention_mask
            )
            logits = model(input_ids, attention_masks=attention_mask)
        except (AttributeError, TypeError):
            # Fallback to using forward with attention output
            outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                logits, label_att = outputs[:2]
            else:
                raise ValueError("Model doesn't provide attention outputs")
    
    # Get label probabilities if needed
    label_probs = None
    if strategy in ["pred_weighted", "predicted_labels"]:
        label_probs = torch.sigmoid(logits)
    
    # If threshold is None and evidence_masks provided, find optimal threshold
    if threshold is None and evidence_masks is not None:
        threshold = find_optimal_attention_threshold(
            label_att=label_att,
            attention_mask=attention_mask,
            evidence_masks=evidence_masks,
            label_probs=label_probs,
            strategy=strategy,
            probability_threshold=probability_threshold,
        )
    elif threshold is None:
        # Default threshold if not provided
        threshold = 0.5
    
    # Build selection mask using build_selection_mask_from_label_attention
    sel_mask = build_selection_mask_from_label_attention(
        label_att=label_att,
        attention_mask=attention_mask,
        strategy=strategy,
        label_probs=label_probs,
        threshold=threshold,
        fallback_top1=fallback_top1,
    )
    
    return sel_mask
