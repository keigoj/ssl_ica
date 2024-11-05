from typing import Optional
from transformers import HubertForCTC
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from attention import MultiHeadAttentionLayer, MultiHeadAttentionLayerForICA


class HubertForCTCWeighted(HubertForCTC):
    def __init__(
            self, 
            config,
            target_lang: Optional[str] = None,
            num_heads: int = 4,
            method: Optional[str] = None,
        ):
        super().__init__(config)

        self.method = method
        self.interctc_idx = [3, 6, 9]

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        if method == "att":
            self.lm_haed_phn = nn.Linear(config.hidden_size, config.phn_vocab_size)
            self.conditioning_layer = nn.Linear(config.vocab_size, config.hidden_size)
            self.conditioning_layer_phn = nn.Linear(config.phn_vocab_size, config.hidden_size)

            self.multihead_attn_low = MultiHeadAttentionLayer(num_heads, config.hidden_size)
            self.multihead_attn_upp = MultiHeadAttentionLayer(num_heads, config.hidden_size)

        elif method == "att_ica":
            self.lm_haed_phn = nn.Linear(config.hidden_size, config.phn_vocab_size)
            self.conditioning_layer = nn.Linear(config.vocab_size, config.hidden_size)
            self.conditioning_layer_phn = nn.Linear(config.phn_vocab_size, config.hidden_size)

            self.multihead_attn_low = MultiHeadAttentionLayerForICA(num_heads, config.hidden_size)
            self.multihead_attn_upp = MultiHeadAttentionLayerForICA(num_heads, config.hidden_size)

    def compute_ctc_loss(self, log_probs, flattened_targets, input_lengths, target_lengths, blank):
        with torch.backends.cudnn.flags(enabled=False):
            loss = nn.functional.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                blank=blank,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )
    
        return loss

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        phn_labels: Optional[torch.Tensor] = None,
        lam: float = 0.5,
        ica_mat: Optional[dict] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")
        if phn_labels.max() >= self.config.phn_vocab_size:
            raise ValueError(f"Phoneme label values must be <= vocab_size: {self.config.phn_vocab_size}")

        # with torch.no_grad():
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # weighted sum
        last_hidden_states = outputs.last_hidden_state
        hidden_states = outputs.hidden_states[1:]
        num_block = len(hidden_states)
        mid = num_block // 2

        if self.method == "att" or self.method == "att_ica":
            # lower layers
            if ica_mat is not None:
                ctc_out_mid = nn.functional.log_softmax(self.lm_haed_phn(hidden_states[mid-1]), dim=-1)
                mid_query = self.conditioning_layer_phn(ctc_out_mid)
                mid_query = mid_query.unsqueeze(1) # (B, 1, T, D)

                mid_query_centerd = mid_query - ica_mat["mean"][mid-1] # (B, 1, T, D) - (1, T, D) -> (B, 1, T, D)
                mid_query_pca = mid_query_centerd @ ica_mat["trans_mat_from_right"][mid-1] # (B, 1, T, D) x (D, C) -> (B, 1, T, C)
                mid_query_ica = mid_query_pca @ ica_mat["W"][mid-1].transpose(0, 1) # (B, 1, T, C)

                lower_hidden_states = torch.stack(hidden_states[:mid], dim=1) # (B, L, T, D)
                lower_centerd = lower_hidden_states - ica_mat["mean"][:mid] # (B, L, T, D) - (L, T, D) -> (B, L, T, D)
                lower_pca = lower_centerd @ ica_mat["trans_mat_from_right"][:mid] # (B, L, T, D) x (B, L, D, C) -> (B, L, T, C)
                lower_ica = lower_pca @ ica_mat["W"][:mid].transpose(-1, -2) # (B, L, T, C)
                lower_weigted = self.multihead_attn_low(mid_query_ica.transpose(1, 2), lower_ica.transpose(1, 2), lower_hidden_states.transpose(1, 2)) # (B, T, D)
                logits_low = self.lm_haed_phn(lower_weigted)
            else:
                ctc_out_mid = nn.functional.log_softmax(self.lm_haed_phn(hidden_states[mid-1]), dim=-1)
                mid_query = self.conditioning_layer_phn(ctc_out_mid)
                mid_query = mid_query.unsqueeze(2) # (B, T, 1, D)

                lower_hidden_states = torch.stack(hidden_states[:mid], dim=2) # (B, T, L, D)
                lower_weigted = self.multihead_attn_low(mid_query, lower_hidden_states, lower_hidden_states) # (B, T, D)
                logits_low = self.lm_haed_phn(lower_weigted)

            # upper layers
            if ica_mat is not None:
                ctc_out_final = nn.functional.log_softmax(self.lm_head(hidden_states[-1]), dim=-1)
                final_query = self.conditioning_layer(ctc_out_final)
                final_query = final_query.unsqueeze(1) # (B, 1, T, D)

                final_query_centerd = final_query - ica_mat["mean"][-1] # (B, 1, T, D) - (1, T, D) -> (B, 1, T, D)
                final_query_pca = final_query_centerd @ ica_mat["trans_mat_from_right"][-1] # (B, 1, T, D) x (D, C) -> (B, 1, T, C)
                final_query_ica = final_query_pca @ ica_mat["W"][mid-1].transpose(0, 1) # (B, 1, T, C)

                upper_hidden_states = torch.stack(hidden_states[mid:], dim=1) # (B, L, T, D)
                upper_centerd = upper_hidden_states - ica_mat["mean"][mid:] # (B, L, T, D) - (L, T, D) -> (B, L, T, D)
                upper_pca = upper_centerd @ ica_mat["trans_mat_from_right"][mid:] # (B, L, T, D) x (B, L, D, C) -> (B, L, T, C)
                upper_ica = upper_pca @ ica_mat["W"][mid:].transpose(-1, -2) # (B, L, T, C)
                upper_weigted = self.multihead_attn_low(final_query_ica.transpose(1, 2), upper_ica.transpose(1, 2), upper_hidden_states.transpose(1, 2)) # (B, T, D)
                logits_upp = self.lm_head(upper_weigted)
            else:
                ctc_out_final = nn.functional.log_softmax(self.lm_head(hidden_states[-1]), dim=-1)
                final_query = self.conditioning_layer(ctc_out_final)
                final_query = final_query.unsqueeze(2) # (B, T, 1, D)
                
                upper_hidden_states = torch.stack(hidden_states[mid:], dim=2) # (B, T, L, D)
                upper_weigted = self.multihead_attn_upp(final_query, upper_hidden_states, upper_hidden_states)
                logits_upp = self.lm_head(upper_weigted)

        last_hidden_states = self.dropout(hidden_states[-1])
        logits = self.lm_head(last_hidden_states)

        loss = None
        low_loss = None
        upp_loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            if self.method == "att" or self.method == "att_ica":
                phn_labels_mask = phn_labels >= 0
                phn_target_lengths = phn_labels_mask.sum(-1)
                phn_flattened_targets = phn_labels.masked_select(phn_labels_mask)

                # ctc_loss doesn't support fp16
                log_probs_low = nn.functional.log_softmax(logits_low, dim=-1, dtype=torch.float32).transpose(0, 1)
                log_probs_upp = nn.functional.log_softmax(logits_upp, dim=-1, dtype=torch.float32).transpose(0, 1)

                low_loss = self.compute_ctc_loss(log_probs_low, phn_flattened_targets, input_lengths, phn_target_lengths, self.config.phn_pad_token_id)
                upp_loss = self.compute_ctc_loss(log_probs_upp, flattened_targets, input_lengths, target_lengths, self.config.pad_token_id)
                
                aux_loss = (low_loss + upp_loss) / 2
            elif self.method == "interctc":
                inter_losses = []
                for idx, state in enumerate(hidden_states):
                    if idx + 1 in self.interctc_idx:
                        inter_out = self.dropout(state)
                        inter_logits = self.lm_head(inter_out)
                        inter_log_probs = nn.functional.log_softmax(inter_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
                        inter_losses.append(self.compute_ctc_loss(inter_log_probs, flattened_targets, input_lengths, target_lengths, self.config.pad_token_id))
                
                aux_loss = sum(inter_losses) / len(inter_losses)

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            std_loss = self.compute_ctc_loss(log_probs, flattened_targets, input_lengths, target_lengths, self.config.pad_token_id)

            loss = (1 - lam) * std_loss + lam * aux_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "aux_loss": aux_loss,
            "low_loss": low_loss,
            "upp_loss": upp_loss,
            "hidden_states": None,
            "attentions": None,
        }