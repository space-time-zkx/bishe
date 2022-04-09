# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import numpy as np
# from utils.misc import NestedTensor
# from utils.pc_utils import shift_scale_points

# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x


class NerfPositionalEncoding(nn.Module):
    def __init__(self, depth=10, sine_type='lin_sine'):
        '''
        out_dim = in_dim * depth * 2
        '''
        super().__init__()
        if sine_type == 'lin_sine':
            self.bases = [i+1 for i in range(depth)]
        elif sine_type == 'exp_sine':
            self.bases = [2**i for i in range(depth)]
        print(f'using {sine_type} as positional encoding')

    @torch.no_grad()
    def forward(self, inputs):
        out = torch.cat([torch.sin(i * math.pi * inputs) for i in self.bases] + [torch.cos(i * math.pi * inputs) for i in self.bases], axis=-1)
        assert torch.isnan(out).any() == False
        return out

# class PositionEmbeddingSine(nn.Module):
#     """
#     This is a more standard version of the position embedding, very similar to the one
#     used by the Attention is all you need paper, generalized to work on images.
#     """
#     def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, sine_type='lin_sine'):
#         super().__init__()
#         self.num_pos_feats = num_pos_feats
#         self.temperature = temperature
#         self.normalize = normalize
#         self.sine = NerfPositionalEncoding(num_pos_feats//2, sine_type)

#     @torch.no_grad()
#     def forward(self, tensor_list: NestedTensor):
#         x = tensor_list.tensors
#         mask = tensor_list.mask
#         assert mask is not None
#         not_mask = ~mask
#         y_embed = not_mask.cumsum(1, dtype=torch.float32)
#         x_embed = not_mask.cumsum(2, dtype=torch.float32)
#         eps = 1e-6
#         y_embed = (y_embed-0.5) / (y_embed[:, -1:, :] + eps)
#         x_embed = (x_embed-0.5) / (x_embed[:, :, -1:] + eps)
#         pos = torch.stack([x_embed, y_embed], dim=-1)
#         return self.sine(pos).permute(0, 3, 1, 2)

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(32, num_pos_feats)
        self.col_embed = nn.Embedding(32, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        # x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

def build_position_encoding(dim):
    # N_steps = args.hidden_dim // 2
    # if args.position_embedding in ('v2', 'sine'):
    #     # TODO find a better way of exposing other arguments
    #     position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    # elif args.position_embedding in ('v3', 'learned'):
    position_embedding = PositionEmbeddingLearned(dim)
    # else:
    #     raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding


# class PositionEmbeddingCoordsSine(nn.Module):
#     def __init__(
#         self,
#         temperature=10000,
#         normalize=False,
#         scale=None,
#         pos_type="fourier",
#         d_pos=None,
#         d_in=3,
#         gauss_scale=1.0,
#     ):
#         super().__init__()
#         self.temperature = temperature
#         self.normalize = normalize
#         if scale is not None and normalize is False:
#             raise ValueError("normalize should be True if scale is passed")
#         if scale is None:
#             scale = 2 * math.pi
#         assert pos_type in ["sine", "fourier"]
#         self.pos_type = pos_type
#         self.scale = scale
#         if pos_type == "fourier":
#             assert d_pos is not None
#             assert d_pos % 2 == 0
#             # define a gaussian matrix input_ch -> output_ch
#             B = torch.empty((d_in, d_pos // 2)).normal_()
#             B *= gauss_scale
#             self.register_buffer("gauss_B", B)
#             self.d_pos = d_pos

#     def get_sine_embeddings(self, xyz, num_channels, input_range):
#         # clone coords so that shift/scale operations do not affect original tensor
#         orig_xyz = xyz
#         xyz = orig_xyz.clone()

#         ncoords = xyz.shape[1]
#         if self.normalize:
#             xyz = shift_scale_points(xyz, src_range=input_range)

#         ndim = num_channels // xyz.shape[2]
#         if ndim % 2 != 0:
#             ndim -= 1
#         # automatically handle remainder by assiging it to the first dim
#         rems = num_channels - (ndim * xyz.shape[2])

#         assert (
#             ndim % 2 == 0
#         ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

#         final_embeds = []
#         prev_dim = 0

#         for d in range(xyz.shape[2]):
#             cdim = ndim
#             if rems > 0:
#                 # add remainder in increments of two to maintain even size
#                 cdim += 2
#                 rems -= 2

#             if cdim != prev_dim:
#                 dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
#                 dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

#             # create batch x cdim x nccords embedding
#             raw_pos = xyz[:, :, d]
#             if self.scale:
#                 raw_pos *= self.scale
#             pos = raw_pos[:, :, None] / dim_t
#             pos = torch.stack(
#                 (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
#             ).flatten(2)
#             final_embeds.append(pos)
#             prev_dim = cdim

#         final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
#         return final_embeds

#     def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
#         # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

#         if num_channels is None:
#             num_channels = self.gauss_B.shape[1] * 2

#         bsize, npoints = xyz.shape[0], xyz.shape[1]
#         assert num_channels > 0 and num_channels % 2 == 0
#         d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
#         d_out = num_channels // 2
#         assert d_out <= max_d_out
#         assert d_in == xyz.shape[-1]

#         # clone coords so that shift/scale operations do not affect original tensor
#         orig_xyz = xyz
#         xyz = orig_xyz.clone()

#         ncoords = xyz.shape[1]
#         if self.normalize:
#             xyz = shift_scale_points(xyz, src_range=input_range)

#         xyz *= 2 * np.pi
#         xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
#             bsize, npoints, d_out
#         )
#         final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

#         # return batch x d_pos x npoints embedding
#         final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
#         return final_embeds

#     def forward(self, xyz, num_channels=None, input_range=None):
#         assert isinstance(xyz, torch.Tensor)
#         assert xyz.ndim == 3
#         # xyz is batch x npoints x 3
#         if self.pos_type == "sine":
#             with torch.no_grad():
#                 return self.get_sine_embeddings(xyz, num_channels, input_range)
#         elif self.pos_type == "fourier":
#             with torch.no_grad():
#                 return self.get_fourier_embeddings(xyz, num_channels, input_range)
#         else:
#             raise ValueError(f"Unknown {self.pos_type}")

#     def extra_repr(self):
#         st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
#         if hasattr(self, "gauss_B"):
#             st += (
#                 f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
#             )
#         return st
