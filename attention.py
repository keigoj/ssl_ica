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