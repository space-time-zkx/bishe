import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# 定义residual
class RB(nn.Module):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1):
        super(RB, self).__init__()
        self.rb = nn.Sequential(nn.Conv1d(nin, nout, ksize, stride, pad),
                                nn.BatchNorm1d(nout),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(nin, nout, ksize, stride, pad),
                                nn.BatchNorm1d(nout))
    def forward(self, input):
        x = input
        x = self.rb(x)

        return nn.ReLU(inplace=False)(input + x)

class SE(nn.Module):
    def __init__(self, nin, nout, reduce=2):
        super(SE, self).__init__()
        self.gp = nn.AdaptiveAvgPool1d(1)
        self.rb1 = RB(nin, nout)
        self.se = nn.Sequential(nn.Linear(nout, nout // reduce),
                                nn.ReLU(inplace=True),
                                nn.Linear(nout // reduce, nout),
                                nn.Sigmoid())
    def forward(self, input):

        x = input
        x = self.rb1(x)

        b, c, p = x.size()
        # print(x.size(),self.gp(x).shape)
        y = self.gp(x).view(b, c,1)

        # print(self.se,y.shape)
        y = self.se(y.permute(0,2,1)).view(b, c, 1)
        # print(y.shape,x.shape)
        y = x * y.expand_as(x)
        out = y + input
        return out

class AdditiveAttention(nn.Module):
    def __init__(self, keys_size, queries_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_q = nn.Linear(queries_size, queries_size, bias=False)
        self.W_k = nn.Linear(keys_size, keys_size, bias=False)
        self.W_v = nn.Linear(queries_size+keys_size,2, bias=False)
        self.dropout = nn.Dropout(dropout)
        # self.norm1 = nn.LayerNorm(queries_size)
        # self.norm2 = nn.LayerNorm(keys_size)
    def forward(self, queries, keys):
        # print(queries.shape,keys.shape,self.W_q,self.W_k)
        queries1 = queries.permute(0,2,1)
        keys1 = keys.permute(0,2,1)
        # queries1, keys1 = self.norm1(self.dropout(F.relu(self.W_q(queries1)))),self.norm1(self.dropout(F.relu(self.W_k(keys1))))
        queries1,keys1 = self.W_q(queries1),self.W_k(keys1)
        '''
        queries --> [batch_size, queries_length, num_hiddens]
        keys --> [batch_size, keys_length, num_hiddens]'''
        # features = queries1.unsqueeze(2) + keys1.unsqueeze(1)
        features = torch.cat([keys1,queries1],dim=2)
        # print(features.shape)
        '''
        queries.unsqueeze(2) --> [batch_size, queries_length, 1, num_hiddens]
        keys.unsqueeze(1) --> [batch_size, 1, keys_length, num_hiddens]
        features --> [batch_size, queries_length,  keys_length, num_hiddens] '''
        features = torch.tanh(features)
        # print(features.shape)
        scores = self.W_v(features)
        # print(scores.shape)
        scores = scores.squeeze(-1)
        '''
        self.W_v(features) --> [batch_size, queries_length,  keys_length, 1]
        scores--> [batch_size, queries_length,  keys_length]'''
        '''
            self.attention_weights --> [batch_size, queries_length,  keys_length]'''
        # print(scores.shape)
        self.attention_weights = F.softmax(scores, dim=2)
        # print(self.attention_weights.shape)
        return self.attention_weights
# class ParallelCoAttentionNetwork(nn.Module):
#
#     def __init__(self, hidden_dim, co_attention_dim, src_length_masking=True):
#         super(ParallelCoAttentionNetwork, self).__init__()
#
#         self.hidden_dim = hidden_dim
#         # self.hidden_dim1 = hidden_dim1
#
#         self.co_attention_dim = co_attention_dim
#         self.src_length_masking = src_length_masking
#
#         self.W_b = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
#         # self.W_b1 = nn.Parameter(torch.randn(self.hidden_dim1, self.hidden_dim1))
#         self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
#         self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
#         # self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
#         # self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))
#         self.W_l = nn.Linear(hidden_dim + hidden_dim, 2, bias=False)
#     def forward(self, V, Q):
#         """
#         :param V: batch_size * hidden_dim * region_num, eg B x 512 x 196
#         :param Q: batch_size * seq_len * hidden_dim, eg B x L x 512
#         :param Q_lengths: batch_size
#         :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
#         batch_size * hidden_dim, batch_size * hidden_dim
#         """
#         # print(V.shape,Q.shape)
#         Q =Q.permute(0,2,1)
#         # (batch_size, seq_len, region_num)
#         C = torch.matmul(Q, torch.matmul(self.W_b, V))
#
#         # (batch_size, co_attention_dim, region_num)
#         H_v = nn.Tanh()(torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
#         # (batch_size, co_attention_dim, seq_len)
#         H_q = nn.Tanh()(
#             torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))
#
#         # (batch_size, 1, region_num)
#         # print(H_v.shape,H_q.shape)
#         features = torch.cat([H_v,H_q],dim=1)
#         features = features.permute(0,2,1)
#         scores = self.W_l(features)
#         scores = scores.squeeze(-1)
#         self.attention_weights = F.softmax(scores, dim=2)
#         # a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)
#         # # (batch_size, 1, seq_len)
#         # a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)
#         # # (batch_size, 1, seq_len)
#
#         # masked_a_q = masked_softmax(
#         #     a_q.squeeze(1), Q_lengths, self.src_length_masking
#         # ).unsqueeze(1)
#
#         # (batch_size, hidden_dim)
#         # v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
#         # (batch_size, hidden_dim)
#         # q = torch.squeeze(torch.matmul(a_q, Q))
#
#         # return a_v, a_q, v, q
#         return self.attention_weights
#
from typing import Dict, Optional

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
def create_src_lengths_mask(
        batch_size: int, src_lengths: Tensor, max_src_len: Optional[int] = None
):
    """
    Generate boolean mask to prevent attention beyond the end of source
    Inputs:
      batch_size : int
      src_lengths : [batch_size] of sentence lengths
      max_src_len: Optionally override max_src_len for the mask
    Outputs:
      [batch_size, max_src_len]
    """
    if max_src_len is None:
        max_src_len = int(src_lengths.max())
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)

    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()


def masked_softmax(scores, src_lengths, src_length_masking=True):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""
    if src_length_masking:
        bsz, max_src_len = scores.size()
        # print('bsz:', bsz)
        # compute masks
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        # Fill pad positions with -inf
        scores = scores.masked_fill(src_mask == 0, -np.inf)

    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)

class ParallelCoAttentionNetwork(nn.Module):

    def __init__(self, hidden_dim, co_attention_dim, src_length_masking=True):
        super(ParallelCoAttentionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.co_attention_dim = co_attention_dim
        self.src_length_masking = src_length_masking

        self.W_b = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1024))
        self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1024))

    def forward(self, V, Q, Q_lengths):
        Q = Q.permute(0,2,1)
        """
        :param V: batch_size * hidden_dim * region_num, eg B x 512 x 196
        :param Q: batch_size * seq_len * hidden_dim, eg B x L x 512
        :param Q_lengths: batch_size
        :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
        batch_size * hidden_dim, batch_size * hidden_dim
        """
        # (batch_size, seq_len, region_num)
        C = torch.matmul(Q, torch.matmul(self.W_b, V))
        # (batch_size, co_attention_dim, region_num)
        H_v = nn.Tanh()(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
        # (batch_size, co_attention_dim, seq_len)
        H_q = nn.Tanh()(
            torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))

        # (batch_size, 1, region_num)
        # print(H_q.shape,H_v.shape)
        a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)
        # (batch_size, 1, seq_len)
        a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)
        # # (batch_size, 1, seq_len)

        masked_a_q = masked_softmax(
            a_q.squeeze(1), Q_lengths, self.src_length_masking
        ).unsqueeze(1)

        # (batch_size, hidden_dim)
        # print(a_v.shape,V.shape,masked_a_q.shape,Q.shape)
        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
        # (batch_size, hidden_dim)
        q = torch.squeeze(torch.matmul(a_q, Q))

        return a_v, a_q, v, q
class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()
    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg

        wei = self.sigmoid(xlg)
        # print(torch.sum(wei)/(torch.sum(wei)+torch.sum(1-wei)),torch.sum(1-wei)/(torch.sum(wei)+torch.sum(1-wei)))
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        print("wei2",torch.sum(wei2)/(torch.sum(wei2)+torch.sum(1-wei2)),torch.sum(1-wei2)/(torch.sum(wei2)+torch.sum(1-wei2)))
        return xo