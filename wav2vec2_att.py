from typing import Optional
from transformers import Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import math


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_feat = n_feat
        self.n_head = n_head
        self.d_head = n_feat // n_head
        assert n_feat % n_head == 0

        self.linear_q = nn.Linear(self.d_head, self.d_head)
        self.linear_k = nn.Linear(self.d_head, self.d_head)
        self.linear_v = nn.Linear(self.d_head, self.d_head)
        self.linear_out = nn.Linear(self.n_feat, self.n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, q, k, v) -> torch.Tensor:
        """multi-head attention

        Args:
            q (torch.Tensor): Size([B, T, 1, D])
            k (torch.Tensor): Size([B, T, L, D])
            v (torch.Tensor): Size([B, T, L, D])

        Returns:
            torch.Tensor: Size([B, T, D])
        """
        assert q.shape[-1] == self.n_feat
        assert k.shape[-1] == self.n_feat
        assert v.shape[-1] == self.n_feat

        q = q.view(*q.shape[:-1], self.n_head, self.d_head)
        k = k.view(*k.shape[:-1], self.n_head, self.d_head)
        v = v.view(*v.shape[:-1], self.n_head, self.d_head)

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # q: (B, T, 1, n_head, d_head) -> (B, T, n_head, 1, d_head)
        q = q.transpose(-2, -3)
        # k: (B, T, L, n_head, d_head) -> (B, T, n_head, L, d_head)
        k = k.transpose(-2, -3)

        # scaled dot-product attention
        # attw: (B, T, n_head, 1, L)
        attw = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)
        self.attn = torch.softmax(attw, dim=-1)
        p_attw = self.dropout(self.attn)

        # v: (B, T, L, n_head, d_head) -> (B, T, n_head, L, d_head)
        v = v.transpose(-2, -3)

        # ct : (B, T, n_head, 1, L) * (B, T, n_head, L, d_head) -> (B, T, n_head, 1, d_head)
        ct = torch.matmul(p_attw, v)

        # ct: (B, T, n_head, 1, d_head) -> (B, T, n_head, d_head)
        ct = ct.squeeze(-2)

        # ct: (B, T, n_head, d_head) -> (B, T, D)
        ct = ct.view(*ct.shape[:-2], self.n_feat)

        return self.linear_out(ct)


class MultiHeadAttentionLayerForICA(MultiHeadAttentionLayer):
    def __init__(
            self, 
            n_head,
            n_feat,
            n_chead=4,
            n_comp=100,
            dropout_rate=0.1
        ):
        super(MultiHeadAttentionLayerForICA, self).__init__(n_head, n_feat, dropout_rate)
        self.n_feat = n_feat
        self.n_head = n_head
        self.d_head = n_feat // n_head
        assert n_feat % n_head == 0

        self.n_comp = n_comp
        self.c_head = n_comp // n_head
        assert n_comp % n_head == 0

        self.linear_q = nn.Linear(self.c_head, self.c_head)
        self.linear_k = nn.Linear(self.c_head, self.c_head)
        self.linear_v = nn.Linear(self.d_head, self.d_head)
        self.linear_out = nn.Linear(self.n_feat, self.n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, q, k, v) -> torch.Tensor:
        """multi-head attention

        Args:
            q (torch.Tensor): Size([B, T, 1, C])
            k (torch.Tensor): Size([B, T, L, C])
            v (torch.Tensor): Size([B, T, L, D])

        Returns:
            torch.Tensor: Size([B, T, D])
        """
        assert q.shape[-1] == self.n_comp
        assert k.shape[-1] == self.n_comp
        assert v.shape[-1] == self.n_feat

        q = q.view(*q.shape[:-1], self.n_head, self.c_head)
        k = k.view(*k.shape[:-1], self.n_head, self.c_head)
        v = v.view(*v.shape[:-1], self.n_head, self.d_head)

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # q: (B, T, 1, n_chead, d_head) -> (B, T, n_chead, 1, c_head)
        q = q.transpose(-2, -3)
        # k: (B, T, L, n_chead, c_head) -> (B, T, n_chead, L, c_head)
        k = k.transpose(-2, -3)

        # scaled dot-product attention
        # attw: (B, T, n_head, 1, L)
        attw = torch.matmul(q, k.transpose(-2, -1)) / (self.c_head**0.5)
        self.attn = torch.softmax(attw, dim=-1)
        p_attw = self.dropout(self.attn)

        # v: (B, T, L, n_head, d_head) -> (B, T, n_head, L, d_head)
        v = v.transpose(-2, -3)

        # ct : (B, T, n_head, 1, L) * (B, T, n_head, L, d_head) -> (B, T, n_head, 1, d_head)
        ct = torch.matmul(p_attw, v)

        # ct: (B, T, n_head, 1, d_head) -> (B, T, n_head, d_head)
        ct = ct.squeeze(-2)

        # ct: (B, T, n_head, d_head) -> (B, T, D)
        ct = ct.view(*ct.shape[:-2], self.n_feat)

        return self.linear_out(ct)


class PredictionModule(nn.Module):
    def __init__(
            self,
            d_model,
            d_ff,
            vocab_size,
            n_head=4,
            dropout_rate=0.1
        ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.multi_head_attn = MultiHeadedAttention(n_head, d_ff, dropout_rate)
        self.ffn1 = nn.Linear(d_ff, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_ff)
        self.linear2 = nn.Linear(d_ff, vocab_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.multi_head_attn(x, x, x, mask=None)
        x = self.ffn2(nn.functional.relu(self.ffn1(x)))
        x = self.linear2(x)

        return x


class Wav2Vec2ForCTCWeighted(Wav2Vec2ForCTC):
    def __init__(
            self, 
            config,
            target_lang: Optional[str] = None,
            num_heads: int = 4,
            ica_path: str = None,
            method: Optional[str] = None,
            two_stage_ft: bool = False,
        ):
        super().__init__(config)

        self.method = method
        self.interctc_idx = [3, 6, 9]

        if two_stage_ft == False:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        if method == "interctc":
            # self.pred_modules = nn.ModuleDict()
            # for idx in self.interctc_idx:
            #     self.pred_modules[f"{idx}"] = PredictionModule(d_model=config.hidden_size, d_ff=512, vocab_size=config.vocab_size, n_head=8)

            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        elif method == "att":
            # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
            self.lm_haed_phn = nn.Linear(config.hidden_size, config.phn_vocab_size)
            self.conditioning_layer = nn.Linear(config.vocab_size, config.hidden_size)
            self.conditioning_layer_phn = nn.Linear(config.phn_vocab_size, config.hidden_size)

            self.multihead_attn_low = MultiHeadAttentionLayer(num_heads, config.hidden_size)
            self.multihead_attn_upp = MultiHeadAttentionLayer(num_heads, config.hidden_size)

            # self.lower_pred_module = PredictionModule(d_model=config.hidden_size, d_ff=512, vocab_size=config.phn_vocab_size, n_head=8)
            # self.upper_pred_module = PredictionModule(d_model=config.hidden_size, d_ff=512, vocab_size=config.vocab_size, n_head=8)
        elif method == "att_ica":
            # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
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
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # weighting hidden states
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
                # ctc_out_mid = nn.functional.log_softmax(self.lower_pred_module.linear2(self.lower_pred_module.linear1(hidden_states[mid-1])), dim=-1)
                ctc_out_mid = nn.functional.log_softmax(self.lm_haed_phn(hidden_states[mid-1]), dim=-1)
                mid_query = self.conditioning_layer_phn(ctc_out_mid)
                mid_query = mid_query.unsqueeze(2) # (B, T, 1, D)

                lower_hidden_states = torch.stack(hidden_states[:mid], dim=2) # (B, T, L, D)
                lower_weigted = self.multihead_attn_low(mid_query, lower_hidden_states, lower_hidden_states) # (B, T, D)
                # logits_low = self.lower_pred_module(lower_weigted)
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
                # ctc_out_final = nn.functional.log_softmax(self.upper_pred_module.linear2(self.upper_pred_module.linear1(hidden_states[-1])), dim=-1)
                ctc_out_final = nn.functional.log_softmax(self.lm_head(hidden_states[-1]), dim=-1)
                final_query = self.conditioning_layer(ctc_out_final)
                final_query = final_query.unsqueeze(2) # (B, T, 1, D)
                
                upper_hidden_states = torch.stack(hidden_states[mid:], dim=2) # (B, T, L, D)
                upper_weigted = self.multihead_attn_upp(final_query, upper_hidden_states, upper_hidden_states)
                # logits_upp = self.upper_pred_module(upper_weigted)
                logits_upp = self.lm_head(upper_weigted)

        last_hidden_states = self.dropout(last_hidden_states)
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
                
                # aux_loss = 0.25*low_loss + 0.75*upp_loss
                aux_loss = (low_loss + upp_loss) / 2
                # aux_loss = upp_loss
            elif self.method == "interctc":
                inter_losses = []
                for idx, state in enumerate(hidden_states):
                    if idx + 1 in self.interctc_idx:
                        inter_out = self.dropout(state)
                        # inter_logits = self.lm_head(inter_out)
                        inter_logits = self.pred_modules[f"{idx+1}"](inter_out)
                        inter_log_probs = nn.functional.log_softmax(inter_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
                        inter_losses.append(self.compute_ctc_loss(inter_log_probs, flattened_targets, input_lengths, target_lengths, self.config.pad_token_id))
                
                aux_loss = sum(inter_losses) / len(inter_losses)

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            std_loss = self.compute_ctc_loss(log_probs, flattened_targets, input_lengths, target_lengths, self.config.pad_token_id)

            loss = (1 - lam) * std_loss + lam * aux_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # return CausalLMOutput(
        #     loss=loss, logits=logits, hidden_states=None, attentions=outputs.attentions
        # )
        return {
            "loss": loss,
            "logits": logits,
            "aux_loss": aux_loss,
            "low_loss": low_loss,
            "upp_loss": upp_loss,
            "hidden_states": None,
            "attentions": None,
        }