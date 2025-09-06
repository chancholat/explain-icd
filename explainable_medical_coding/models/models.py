from typing import Callable, Optional

import torch
import torch.utils.checkpoint
from pydantic import BaseModel
from torch import nn
from transformers import AutoConfig, AutoModel
import sys
sys.path.append("./")
from explainable_medical_coding.explainability.helper_functions import (
    create_baseline_input,
)
from explainable_medical_coding.models.modules.attention import (
    InputMasker,
    LabelAttention,
    LabelCrossAttention,
    InvertLabelCrossAttention
)


class ModuleNames(BaseModel):
    ln_1: str
    ln_2: str
    dense_values: str
    dense_heads: str
    model_layer_name: str


class RobertaModuleNames(ModuleNames):
    ln_1: str = "attention.output.LayerNorm"
    ln_2: str = "output.LayerNorm"
    dense_values: str = "attention.self.value"
    dense_heads: str = "attention.output.dense"
    model_layer_name: str = "roberta_encoder.encoder.layer"


class PLMICD(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_path: str,
        chunk_size: int,
        pad_token_id: int,
        cross_attention: bool = True,
        scale: float = 1.0,
        mask_input: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.pad_token_id = pad_token_id
        self.module_names = RobertaModuleNames()
        self.gradient = None

        self.config = AutoConfig.from_pretrained(
            model_path, num_labels=num_classes, finetuning_task=None
        )

        self.roberta_encoder = AutoModel.from_pretrained(
            model_path, config=self.config, add_pooling_layer=False
        )

        if cross_attention:
            self.label_wise_attention = LabelCrossAttention(
                input_size=self.config.hidden_size, num_classes=num_classes, scale=scale
            )
        else:
            self.label_wise_attention = LabelAttention(
                input_size=self.config.hidden_size,
                projection_size=self.config.hidden_size,
                num_classes=num_classes,
            )
        
        self.invert_label_wise_attention = InvertLabelCrossAttention(
            label_wise_attention=self.label_wise_attention, num_classes=num_classes, scale=scale
        )

        self.mask_input = mask_input
        if self.mask_input:
            self.input_masker = InputMasker(
                input_size=self.config.hidden_size, scale=scale
            )

    @torch.no_grad()
    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward(input_ids=input_ids, attention_masks=attention_mask)
        )

    def split_input_into_chunks(
        self, input_sequence: torch.Tensor, pad_index: int
    ) -> torch.Tensor:
        """Split input into chunks of chunk_size.

        Args:
            input_sequence (torch.Tensor): input sequence to split (batch_size, seq_len)
            pad_index (int): padding index

        Returns:
            torch.Tensor: reshaped input (batch_size, num_chunks, chunk_size)
        """
        batch_size = input_sequence.size(0)
        # pad input to be divisible by chunk_size
        input_sequence = nn.functional.pad(
            input_sequence,
            (0, self.chunk_size - input_sequence.size(1) % self.chunk_size),
            value=pad_index,
        )
        return input_sequence.view(batch_size, -1, self.chunk_size)

    def roberta_encode_embedding_input(self, embedding, attention_masks):
        input_shape = embedding.size()[:-1]
        extended_attention_mask = self.roberta_encoder.get_extended_attention_mask(
            attention_masks, input_shape
        )
        head_mask = self.roberta_encoder.get_head_mask(
            None, self.roberta_encoder.config.num_hidden_layers
        )
        encoder_outputs = self.roberta_encoder.encoder(
            embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]
        return sequence_output

    def get_chunked_attention_masks(
        self, attention_masks: torch.Tensor
    ) -> torch.Tensor:
        return self.split_input_into_chunks(attention_masks, 0)

    def get_input_embeddings(self):
        return self.roberta_encoder.embeddings

    def get_chunked_embedding(self, input_ids):
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        chunk_size = input_ids.size(-1)
        embedding = self.roberta_encoder.embeddings(input_ids.view(-1, chunk_size))
        return embedding

    def get_token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings. Huggingface Roberta model can't return more than 512 token embeddings at once.

        Args:
            input_ids (torch.Tensor): input ids

        Returns:
            torch.Tensor: token embeddings
        """
        sequence_length = input_ids.size(1)
        batch_size = input_ids.size(0)
        if sequence_length <= 512:
            return self.roberta_encoder.embeddings(input_ids)
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        chunk_size = input_ids.size(-1)
        chunked_embeddings = self.roberta_encoder.embeddings(
            input_ids.view(-1, chunk_size)
        )
        embeddings = chunked_embeddings.view(
            batch_size, -1, chunked_embeddings.size(-1)
        )
        return embeddings[:, :sequence_length]

    def encoder(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        if attention_masks is not None:
            attention_masks = self.get_chunked_attention_masks(attention_masks)
        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta_encoder(
            input_ids=input_ids.view(-1, chunk_size),
            attention_mask=attention_masks.view(-1, chunk_size)
            if attention_masks is not None
            else None,
            return_dict=False,
        )
        return outputs[0].view(batch_size, num_chunks * chunk_size, -1)

    def forward_with_input_masking(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_mask: bool = False,
        baseline_token_id: int = 500001,
    ):
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=baseline_token_id,
            eos_token_id=baseline_token_id,
        )
        baseline = baseline.requires_grad_(False)
        baseline_embeddings = self.get_chunked_embedding(baseline).detach()
        chunked_input_embeddings = self.get_chunked_embedding(input_ids)
        with torch.no_grad():
            token_representations = (
                self.get_token_representations_from_chunked_embeddings(
                    chunked_input_embeddings.detach(), attention_masks
                )
            )
        input_mask = self.input_masker(
            token_representations, attention_masks=attention_masks
        )

        input_mask_sigmoid = torch.sigmoid(input_mask)
        input_mask_sigmoid = input_mask_sigmoid.view(-1, self.chunk_size, 1)
        masked_chunked_input_embeddings = (
            chunked_input_embeddings * input_mask_sigmoid
            + baseline_embeddings * (1 - input_mask_sigmoid)
        )
        masked_token_representations = (
            self.get_token_representations_from_chunked_embeddings(
                masked_chunked_input_embeddings, attention_masks
            )
        )
        if output_mask:
            return self.label_wise_attention(
                masked_token_representations,
                attention_masks=attention_masks,
                output_attention=output_attentions,
            ), input_mask

        return self.label_wise_attention(
            masked_token_representations,
            attention_masks=attention_masks,
            output_attention=output_attentions,
        )

    def get_token_representations_from_chunked_embeddings(
        self,
        chunked_embedding: torch.Tensor,
        attention_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Get token representations from chunked embeddings.

        Args:
            chunked_embedding (torch.Tensor): Chunked embedding of shape [batch_size*num_chunks, chunk_size, embedding_size]
            attention_masks (torch.Tensor): Attention masks of shape [batch_size, sequence_length]

        Returns:
            torch.Tensor: Token representations of shape [batch_size, sequence_length, hidden_size]
        """
        (
            num_chunks_times_batch_size,
            chunk_size,
            embedding_size,
        ) = chunked_embedding.size()
        batch_size = attention_masks.size(0)
        num_chunks = num_chunks_times_batch_size // batch_size
        chunked_attention_masks = self.get_chunked_attention_masks(attention_masks)
        hidden_outputs = self.roberta_encode_embedding_input(
            embedding=chunked_embedding.view(-1, chunk_size, embedding_size),
            attention_masks=chunked_attention_masks.view(-1, chunk_size),
        )
        return hidden_outputs.view(batch_size, num_chunks * chunk_size, -1)

    def forward_embedding_input(
        self,
        chunked_embedding: torch.Tensor,
        attention_masks: torch.Tensor,
        output_attention: bool = False,
    ) -> torch.Tensor:
        """Forward pass for the model with chunked embedding input.

        Args:
            chunked_embedding (torch.Tensor): Chunked embedding of shape [batch_size*num_chunks, chunk_size, embedding_size]
            attention_masks (torch.Tensor): Attention masks of shape [batch_size, num_chunks, chunk_size]

        Returns:
            torch.Tensor:
        """
        token_representations = self.get_token_representations_from_chunked_embeddings(
            chunked_embedding, attention_masks
        )
        return self.label_wise_attention(
            token_representations,
            attention_masks=attention_masks,
            output_attention=output_attention,
        )

    @torch.no_grad()
    def de_chunk_attention(
        self,
        attentions_chunked: torch.Tensor,
        batch_size: int,
        num_layers: int,
        num_chunks: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """De-chunk attention.

        Args:
            attentions_chunked (torch.Tensor): Attention matrix of shape [batch_size, num_chunks, num_layers, chunk_size, chunk_size]
            batch_size (int): Batch size
            num_layers (int): Number of layers
            num_chunks (int): Number of chunks
            chunk_size (int): Chunk size

        Returns:
            torch.Tensor: Attention matrix of shape [batch_size, num_layers, num_chunks*chunk_size, num_chunks*chunk_size]
        """
        attentions = torch.zeros(
            batch_size,
            num_layers,
            num_chunks * chunk_size,
            num_chunks * chunk_size,
            device=attentions_chunked.device,
            dtype=torch.float16,
        )
        for chunk_idx in range(num_chunks):
            attentions[
                :,
                :,
                chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size,
                chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size,
            ] = attentions_chunked[:, chunk_idx]

        return attentions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.mask_input:
            return self.forward_with_input_masking(
                input_ids, attention_masks, output_attentions, False
            )
        hidden_output = self.encoder(input_ids, attention_masks)
        return self.label_wise_attention(
            hidden_output,
            attention_masks=attention_masks,
            output_attention=output_attentions,
            attn_grad_hook_fn=attn_grad_hook_fn,
        )

    @torch.no_grad()
    def get_encoder_attention_and_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        if attention_masks is not None:
            attention_masks_chunks = self.split_input_into_chunks(attention_masks, 0)
        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta_encoder(
            input_ids=input_ids.view(-1, chunk_size),
            attention_mask=attention_masks_chunks.view(-1, chunk_size)
            if attention_masks_chunks is not None
            else None,
            return_dict=False,
            output_attentions=True,
            output_hidden_states=True,
        )

        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        _, label_wise_attention = self.label_wise_attention(
            hidden_output, attention_masks, True
        )
        return outputs[2], outputs[3], label_wise_attention

    @torch.no_grad()
    def attention_rollout(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            _,
            attentions,
            label_wise_attention,
        ) = self.get_encoder_attention_and_hidden_states(input_ids, attention_masks)

        label_wise_attention = torch.softmax(label_wise_attention, dim=1)
        attentions = (
            torch.stack(attentions).to(torch.float16).transpose(1, 0)
        )  # [batch_size*num_chunks, num_layers, num_heads, chunk_size, chunk_size]

        batch_size = input_ids.size(0)
        num_chunks = attentions.size(0) // batch_size
        num_layers = attentions.size(1)
        num_heads = attentions.size(2)
        chunk_size = attentions.size(3)

        attentions = attentions.view(
            batch_size, num_chunks, num_layers, num_heads, chunk_size, chunk_size
        )
        attentions = torch.mean(
            attentions, dim=3
        )  # [batch_size, num_chunks, num_layers, chunk_size, chunk_size]
        attentions = self.de_chunk_attention(
            attentions, batch_size, num_layers, num_chunks, chunk_size
        )  # [batch_size, num_layers, num_chunks*chunk_size, num_chunks*chunk_size]

        attentions = (
            attentions
            + torch.eye(chunk_size * num_chunks, device=attentions.device)
            .unsqueeze(0)
            .unsqueeze(0)
            / 2
        )  # add skip connection

        attention_rollout = attentions[:, 0]
        for hidden_layer_idx in range(1, num_layers):
            attention_rollout = attentions[:, hidden_layer_idx] @ attention_rollout

        return label_wise_attention @ attention_rollout
    
    def get_invert_label_attention(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_output = self.encoder(input_ids, attention_masks)
        return self.invert_label_wise_attention.forward(hidden_output, attention_masks)

    def forward_with_selected_tokens(
            self,
            input_ids: torch.Tensor,
            attention_masks: Optional[torch.Tensor] = None,
            selected_token_mask: Optional[torch.Tensor] = None,
            stop_gradient_unselected: bool = True,
            return_token_logits: bool = True,
            output_attentions: bool = False,
            mask_pooling: bool = True,                      # NEW: control whether to mask pooling
            soft_alpha: float = 0.0,                        # NEW: soft weight for unselected tokens in pooling (0.0 = hard mask)
            fallback_to_full_attention_if_empty: bool = True,  # NEW: if a sample selects nothing, keep original attention for that sample
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            Encoder once; optionally block gradients for unselected tokens; optionally mask pooling.
            - If mask_pooling=True and soft_alpha==0.0, only selected tokens contribute to pooling.
            - If mask_pooling=True and 0<soft_alpha<1, unselected tokens get weight=soft_alpha at pooling.
            - If mask_pooling=False, pooling uses the original attention mask (full context),
              but you can still stop gradients on unselected tokens.
            """
            token_reps = self.encoder(input_ids, attention_masks)  # (B, L, H)
            # print("token reps:", token_reps.shape)
            eff_attention = attention_masks  # (B, L)
            if selected_token_mask is not None:
                sel = selected_token_mask.to(token_reps.dtype)
                pad_len = token_reps.size(1) - sel.size(1)
                if pad_len > 0:
                    pad_sel = torch.nn.functional.pad(sel, (0, pad_len), value=0)
                # print("sel:", sel.shape)
                # print("sel unsqueeeze(-1):", sel.unsqueeze(-1).shape)
                sel = sel[:, : token_reps.size(1)]  # safety clip
                B, L = sel.shape

                # Row mask: which samples have at least one selected token
                has_any = (sel.sum(dim=1) > 0).to(token_reps.dtype).view(B, 1, 1)  # (B,1,1)
                # print("has any:", has_any.shape)


                # Optional stop-gradient on unselected tokens (only for rows that have selections)
                if stop_gradient_unselected:
                    token_reps = token_reps * has_any + token_reps * (1.0 - has_any)  # no-op to ensure shape
                    token_reps = token_reps * pad_sel.unsqueeze(-1) + token_reps.detach() * (1.0 - pad_sel.unsqueeze(-1))

                if mask_pooling:
                    if soft_alpha <= 0.0:
                        # Hard mask: selected=1, unselected=0
                        ea = (attention_masks.to(sel.dtype) * sel)
                        if fallback_to_full_attention_if_empty:
                            # For rows with no selections, keep the original attention
                            no_sel_rows = (sel.sum(dim=1) == 0)
                            if no_sel_rows.any():
                                ea[no_sel_rows] = attention_masks[no_sel_rows].to(sel.dtype)
                        eff_attention = ea.to(attention_masks.dtype)
                    else:
                        # Soft mask: selected=1.0, unselected=alpha
                        soft_w = soft_alpha + (1.0 - soft_alpha) * sel  # (B, L)
                        if fallback_to_full_attention_if_empty:
                            no_sel_rows = (sel.sum(dim=1) == 0)
                            if no_sel_rows.any():
                                soft_w[no_sel_rows] = soft_alpha   # keep original attention for those rows
                        ea = (attention_masks.to(soft_w.dtype) * soft_w)
                        eff_attention = ea.to(attention_masks.dtype)
                else:
                    # No pooling mask: use original attention (full context)
                    eff_attention = attention_masks

            # Document-level logits via label-wise attention
            doc_logits = self.label_wise_attention(
                token_reps,
                attention_masks=eff_attention,
                output_attention=output_attentions,
            )

            tok_logits = None
            if return_token_logits:
                device = token_reps.device
                if not hasattr(self, "token_aux_head") or self.token_aux_head is None:
                    self.token_aux_head = nn.Linear(token_reps.size(-1), self.num_classes).to(device)
                else:
                    self.token_aux_head = self.token_aux_head.to(device)
                tok_logits = self.token_aux_head(token_reps)  # (B, L, C)

            return doc_logits, tok_logits
