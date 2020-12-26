# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoswiseFeedForwardNet(nn.Module):
  def __init__(self, d_hidn):
    super().__init__()
    self.d_hidn = d_hidn

    self.conv1 = nn.Conv1d(in_channels=d_hidn, out_channels=d_hidn*4, kernel_size=1)
    self.conv2 = nn.Conv1d(in_channels=d_hidn*4, out_channels=d_hidn, kernel_size=1)
    self.active = F.gelu

  def forward(self, inputs):
    output = self.active(self.conv1(inputs.transpose(1,2)))
    output = self.conv2(output).transpose(1,2)
    return output


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_head):

        super().__init__()
        self.scale = 1 / (d_head **0.5)

    def forward(self, Q, K, V, attn_mask):

        scores = torch.matmul(Q, K.transpose(-1,-2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        attn_prob = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn_prob, V)

        return context, attn_prob


class MultiHeadAttention(nn.Module):

    def __init__(self, d_hidn, n_head, d_head):

        super().__init__()
        self.d_hidn = d_hidn
        self.n_head = n_head
        self.d_head = d_head

        self.W_Q = nn.Linear(d_hidn, n_head * d_head)
        self.W_K = nn.Linear(d_hidn, n_head * d_head)
        self.W_V = nn.Linear(d_hidn, n_head * d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(d_head)
        self.linear = nn.Linear(n_head * d_head, d_hidn)
    
    def forward(self, Q, K, V, attn_mask):

        batch_size = Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_head)
        output = self.linear(context)

        return output, attn_prob