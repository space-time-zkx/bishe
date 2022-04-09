import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class SE(nn.Module):
    def __init__(self, nin, nout, reduce=2):
        super(SE, self).__init__()
        self.gp = nn.AvgPool2d(1)
        self.rb1 = RB(nin, nout)
        self.se = nn.Sequential(nn.Linear(nout, nout // reduce),
                                nn.ReLU(inplace=True),
                                nn.Linear(nout // reduce, nout),
                                nn.Sigmoid())
    def forward(self, input):
        x = input
        x = self.rb1(x)

        b, c, _, _ = x.size()
        y = self.gp(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        out = y + input
        return out

class AdditiveAttention(nn.Module):
    def __init__(self, keys_size, queries_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_q = nn.Linear(queries_size, queries_size, bias=False)
        self.W_k = nn.Linear(keys_size, keys_size, bias=False)
        self.W_v = nn.Linear(queries_size+keys_size,1, bias=False)
        self.W_v2 = nn.Linear(queries_size+keys_size,queries_size+keys_size,bias=False)
        self.class1 = nn.Linear(3,queries_size+keys_size,bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, one_hot_vec):
        # print(queries.shape,keys.shape)
        queries = queries.permute(0,2,1)
        keys = keys.permute(0,2,1)
        queries1, keys1 = self.W_q(queries), self.W_k(keys)
        '''
        queries --> [batch_size, queries_length, num_hiddens]
        keys --> [batch_size, keys_length, num_hiddens]'''
        # features = queries1.unsqueeze(2) + keys1.unsqueeze(1)
        # print(keys1.shape,queries1.shape)
        features = torch.cat([keys,queries1],dim=2)
        # print(features.shape)
        '''
        queries.unsqueeze(2) --> [batch_size, queries_length, 1, num_hiddens]
        keys.unsqueeze(1) --> [batch_size, 1, keys_length, num_hiddens]
        features --> [batch_size, queries_length,  keys_length, num_hiddens] '''
        classweight = self.class1(one_hot_vec)

        classweight = classweight.unsqueeze(1)
        featuresepa = torch.cat([torch.zeros(features.shape[0],features.shape[1],512),torch.ones(features.shape[0],features.shape[1],128)],dim=2).cuda()
        featuresepa = self.W_v2(featuresepa)
        features = torch.tanh(features*classweight*featuresepa)
        # print(features.shape)

        # scores = self.W_v(features)
        # print(scores.shape)
        # scores = scores.squeeze(-1)
        '''
        self.W_v(features) --> [batch_size, queries_length,  keys_length, 1]
        scores--> [batch_size, queries_length,  keys_length]'''
        '''
            self.attention_weights --> [batch_size, queries_length,  keys_length]'''
        # print(scores.shape)
        # self.attention_weights = F.softmax(scores, dim=2)
        # print(self.attention_weights.shape)
        return features

        # return torch.bmm(self.dropout(self.attention_weights), values)
    # def forward(self, queries, keys, values):
    #     queries1, keys1 = self.W_q(queries), self.W_k(keys)
    #     '''
    #     queries --> [batch_size, queries_length, num_hiddens]
    #     keys --> [batch_size, keys_length, num_hiddens]'''
    #     features = queries1.unsqueeze(2) + keys1.unsqueeze(1)
    #     '''
    #     queries.unsqueeze(2) --> [batch_size, queries_length, 1, num_hiddens]
    #     keys.unsqueeze(1) --> [batch_size, 1, keys_length, num_hiddens]
    #     features --> [batch_size, queries_length,  keys_length, num_hiddens] '''
    #     features = torch.tanh(features)
    #     scores = self.W_v(features)
    #     # print(scores.shape)
    #     scores = scores.squeeze(-1)
    #     '''
    #     self.W_v(features) --> [batch_size, queries_length,  keys_length, 1]
    #     scores--> [batch_size, queries_length,  keys_length]'''
    #     self.attention_weights = F.softmax(scores, dim=1)
    #     '''
    #     self.attention_weights --> [batch_size, queries_length,  keys_length]'''
    #     return torch.bmm(self.dropout(self.attention_weights), values)
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512,128,bias=False)
    def forward(self, queries, keys, values):
        '''
        queries --> [batch_size, queries_length, queries_feature_num]
          keys --> [batch_size, keys_values_length, keys_features_num]
          values --> [barch_size, keys_values_length, values_features_num]
          点积模型中： queries_features_num = keys_features_num
        '''
        keys = keys.permute(0,2,1)
        keys = self.fc1(keys)
        # queries = queries.unsqueeze(1)
        # keys = keys.unsqueeze(1)
        # values = values.unsqueeze(1)
        # print(queries.shape,keys.shape,values.shape)
        d = keys.shape[-1]
        queries = queries.permute(0,2,1)
        values = values.permute(0,2,1)
        '''交换keys的后两个维度，相当于公式中的转置'''
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = F.softmax(scores, dim=1)
        return torch.bmm(self.dropout(self.attention_weights), values)

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        channel_weights = self.channel_attention(x)

        out = channel_weights * x
        out = self.spatial_attention(out) * out
        # out = F.avg_pool2d(out, [32, 32])
        return out

class CoAttention(nn.Module):
    def __init__(self):
        super(CoAttention, self).__init__()

    def forward(self,x):
        return x
class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        self.same = nn.Conv2d(128,512,kernel_size=1,stride=1,padding=1)
    def forward(self, x, residual):
        if residual.shape[1]!=x.shape[1]:
            residual = self.same(residual)
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            # 给需要mask的地方设置一个负无穷
        	attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):

    def __init__(self,query_size=128,key_size=512,model_dim=512, num_heads=2, dropout=0.3):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(key_size, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(query_size, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(query_size, self.dim_per_head * num_heads)

        # self.W_q = nn.Linear(queries_size, self.dim_per_head * num_heads, bias=False)
        # self.W_k = nn.Linear(keys_size, self.dim_per_head * num_heads, bias=False)
        # self.W_v = nn.Linear(self.dim_per_head * num_heads, 1, bias=False)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, query_size)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(query_size)
    def forward(self, key, value, query, attn_mask=None):
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


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

    def __init__(self, hidden_dim,hidden_dim1,  co_attention_dim, src_length_masking=True):
        super(ParallelCoAttentionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.hidden_dim1 = hidden_dim1
        self.co_attention_dim = co_attention_dim
        self.src_length_masking = src_length_masking

        self.W_b = nn.Parameter(torch.randn(self.hidden_dim1, self.hidden_dim))
        self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim1))
        self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
        self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))

    def forward(self, V, Q, Q_lengths):
        """
        :param V: batch_size * hidden_dim * region_num, eg B x 512 x 196
        :param Q: batch_size * seq_len * hidden_dim1, eg B x L x 512
        :param Q_lengths: batch_size
        :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
        batch_size * hidden_dim, batch_size * hidden_dim
        """
        # (batch_size, seq_len, region_num)
        Q = Q.permute(0,2,1)
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

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
        # (batch_size, hidden_dim)
        q = torch.squeeze(torch.matmul(a_q, Q))

        return a_v, a_q, v, q
# class PAM_Module(Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = Parameter(torch.zeros(1))
#
#         self.softmax = Softmax(dim=-1)
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)
#
#         out = self.gamma*out + x
#         return out
#
#
# class CAM_Module(Module):
#     """ Channel attention module"""
#     def __init__(self, in_dim):
#         super(CAM_Module, self).__init__()
#         self.chanel_in = in_dim
#
#
#         self.gamma = Parameter(torch.zeros(1))
#         self.softmax  = Softmax(dim=-1)
#     def forward(self,x,y):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = x.view(m_batchsize, C, -1)
#         proj_key = y.view(m_batchsize, C, -1).permute(0, 2, 1)
#         energy = torch.bmm(proj_query, proj_key)
#         energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
#         attention = self.softmax(energy_new)
#         proj_value = x.view(m_batchsize, C, -1)
#
#         out = torch.bmm(attention, proj_value)
#         out = out.view(m_batchsize, C, height, width)
#
#         out = self.gamma*out + x
#         return out


