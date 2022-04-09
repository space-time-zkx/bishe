# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from lib.backbones.plainnet import PlainNet
# from lib.backbones.senet import senet18_patchnet as senet
# from lib.backbones.resnet import resnet18_patchnet as resnet
# from lib.backbones.resnext import resnext_patchnet_1 as resnext
# from lib.helpers.fpointnet_helper import parse_outputs
# from lib.extensions.mask_global_pooling import mask_global_max_pooling_2d
# from lib.extensions.mask_global_pooling import mask_global_avg_pooling_2d
# from lib.helpers.misc_helper import init_weights
# import os 
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
# # os.environ["CUDA_VISIBLE_DEVICES"]="1"
# #VIT
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return self.net(x)
    
# class Attention(nn.Module):
#     def __init__(self, dim, heads, dim_head = 64,  dropout=0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#     def forward(self,x):
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head,  dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ]))
#     def forward(self, x):
#         # x = x.squeeze(2)
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x
# class CT(nn.Module):
#     def __init__(self, num_patches, dim,depth, heads, mlp_dim, pool = 'mean', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
#         super().__init__()
#         self.num_patches = num_patches
        
#         patch_dim = channels * self.num_patches
#         # print(patch_dim)
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b n h w -> b n (h w)')
#                         )
#         self.to_patch_embedding2 = nn.Linear(patch_dim, dim)
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#         self.pool = pool
#         self.to_latent = nn.Identity()

#         self.mlp_head = nn.LayerNorm(dim)

#     def forward(self, data, mask = None):
#         x = data
#         # print(x.shape)
#         x = self.to_patch_embedding(x) # x.shape=[b,49,128]
#         # print(x.shape)
#         x = self.to_patch_embedding2(x)
#         # print(x.shape)
       
#         b, n, _ = x.shape # n = 49
#         x += self.pos_embedding[:, :n] # x.shape=[b,50,128]
#         x = self.dropout(x) 

#         x = self.transformer(x) # x.shape=[b,50,128],mask=None
#         # print("o",x.shape)
#         x = x.mean(dim = 2) if self.pool == 'mean' else x[:, 0]

#         x = self.to_latent(x)
#         return self.mlp_head(x)
# class ChannelAttentionModule(nn.Module):
#     def __init__(self, channel, ratio=16):
#         super(ChannelAttentionModule, self).__init__()
#         #使用自适应池化缩减map的大小，保持通道不变
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
 
#         self.shared_MLP = nn.Sequential(
#             nn.Conv2d(channel, channel // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):
#         avgout = self.shared_MLP(self.avg_pool(x))
#         maxout = self.shared_MLP(self.max_pool(x))
#         return self.sigmoid(avgout + maxout)
 
# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):
#         #map尺寸不变，缩减通道
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv2d(out))
#         return out
 
# class CBAM(nn.Module):
#     def __init__(self, channel):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttentionModule(channel)
#         self.spatial_attention = SpatialAttentionModule()
 
#     def forward(self, x):
#         out = self.channel_attention(x) * x
#         out = self.spatial_attention(out) * out
#         # out = F.avg_pool2d(out, [32, 32])
#         return out
# class AM(nn.Module):
#     def __init__(self):
#         super(AM, self).__init__()
#         # self.channel_attention = ChannelAttentionModule(channel)
#         self.spatial_attention = SpatialAttentionModule()
 
#     def forward(self, x):
#         # out = self.channel_attention(x) * x
#         out = self.spatial_attention(x) * x
#         # out = F.avg_pool2d(out, [32, 32])
#         return out
# class PatchNet(nn.Module):
#     def __init__(self, cfg, num_heading_bin, num_size_cluster, mean_size_arr):
#         super().__init__()
#         self.cfg = cfg
#         self.num_heading_bin = num_heading_bin
#         self.num_size_cluster = num_size_cluster
#         self.mean_size_arr = mean_size_arr

#         # center estimation module
#         self.center_reg_backbone = PlainNet(input_channels=3, layer_cfg=[128, 128, 256], kernal_size=1)
#         self.center_reg_head = nn.Sequential(nn.Linear(259, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
#                                              nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
#                                              nn.Linear(128, 3))
#         # box estiamtion module
#         assert cfg['backbone'] in ['plainnet', 'resnet', 'resnext', 'senet']
#         if cfg['backbone'] == 'plainnet':
#             self.box_est_backbone = PlainNet(input_channels=256, layer_cfg=[128, 128, 256], kernal_size=3, padding=1)
#             self.box_est_backbone2 = PlainNet(input_channels=256, layer_cfg=[512], kernal_size=3, padding=1)
#         if cfg['backbone'] == 'resnet':
#             self.box_est_backbone = resnet()
#         if cfg['backbone'] == 'senet':
#             self.box_est_backbone = senet()
#         if cfg['backbone'] == 'resnext':
#             self.box_est_backbone = resnext()

#         self.box_est_head1 = nn.Sequential(nn.Linear(259, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
#                                           nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
#                                           nn.Linear(128, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))
#         self.box_est_head2 = nn.Sequential(nn.Linear(643, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
#                                           nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
#                                           nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))
        
#         self.box_est_head3 = nn.Sequential(nn.Linear(643, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
#                                           nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
#                                           nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))
#         self.box_est_head4 = nn.Sequential(nn.Linear(643, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
#                                           nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
#                                           nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))

#         self.img_backbone = resnet()
#         self.attention2 = AM()
#         self.attention = CBAM(channel=640)
#         # self.transformer = CT(1024,dim=640,depth=3,heads=3,mlp_dim=1280,dropout=0.1,emb_dropout=0.1)
#         init_weights(self, self.cfg['init'])


#     def forward(self, patch1, one_hot_vec):
#         output_dict = {}
#         patch = patch1[:,:3,:,:]
#         # get [h, w] of input patch  [for global pooling]
#         _, _, h, w = patch.shape

#         # mask generation
#         rgb = patch1[:,3:,:,:]
#         # import cv2 
#         # print(rgb[0].permute(1,2,0).detach().cpu().numpy(),rgb[0].permute(1,2,0).detach().cpu().numpy().shape)
#         # cv2.imwrite("test.png",rgb[0].permute(1,2,0).detach().cpu().numpy())
#         depth_map = patch[:, 2, :, :]
#         # print("0",patch.shape)
#         threshold = depth_map.mean(-1).mean(-1) + self.cfg['threshold_offset']
#         threshold = threshold.unsqueeze(-1).unsqueeze(-1).repeat(1, h, w)
#         zeros, ones = torch.zeros_like(depth_map), torch.ones_like(depth_map)
#         mask = torch.where(depth_map < threshold, ones, zeros)
#         mask_xyz_mean = mask_global_avg_pooling_2d(patch, mask)
#         patch = patch - mask_xyz_mean
#         mask_xyz_mean = mask_xyz_mean.squeeze(-1).squeeze(-1)

#         first_features = self.center_reg_backbone(patch)
#         # box_est_features = self.box_est_backbone(first_features)
        
#         # center regressor
#         center_reg_features = mask_global_max_pooling_2d(first_features, mask)
#         center_reg_features = torch.cat([center_reg_features.view(-1, 256), one_hot_vec], -1)  # add one hot vec
#         center_tnet = self.center_reg_head(center_reg_features)
#         stage1_center = center_tnet + mask_xyz_mean  # Bx3
#         output_dict['stage1_center'] = stage1_center

#         # first_features = self.attention2(first_features)
#         # first_features = mask_global_max_pooling_2d(first_features, mask)
#         # first_features = torch.cat([first_features.view(-1, 256), one_hot_vec], -1) 
#         # box1 = self.box_est_head1(first_features)

#         box_est_features1 = self.box_est_backbone(patch)
#         box_est_features = self.box_est_backbone2(box_est_features1)
#         # first_features = self.attention2(first_features)
#         box_est_features1 = mask_global_max_pooling_2d(box_est_features1, mask)
#         box_est_features1 = torch.cat([box_est_features1.view(-1, 256), one_hot_vec], -1) 
#         box1 = self.box_est_head1(box_est_features1)
#         # get patch in object coordinate
#         patch = patch - center_tnet.unsqueeze(-1).unsqueeze(-1)

#         # 3d box regressor
#         # box_est_features = self.box_est_backbone(patch)

#         rgb_features = self.img_backbone(rgb)
#         # print(mask.shape)
#         # rgb_features = mask_global_max_pooling_2d(rgb_features,mask)
#         # rgb_features = torch.cat([rgb_features.view(-1, 256)], -1)
#         # print("1",box_est_features.shape)
#         # box_est_features = mask_global_max_pooling_2d(box_est_features, mask)
#         # box_est_features = torch.cat([box_est_features.view(-1, 512)], -1)  
#         box_est_features = torch.cat([box_est_features,rgb_features],1)
#         box_est_featuresa = self.attention(box_est_features)
#         # print(box_est_features.shape)
#         # box_est_features = box_est_features.reshape((box_est_features.shape[0],box_est_features.shape[1],1,1))
#         # print(box_est_features.shape)
#         # box_est_features = self.transformer(box_est_features)
#         box_est_features = mask_global_max_pooling_2d(box_est_features,mask)
#         box_est_features = torch.cat([box_est_features.view(-1, 640),one_hot_vec], -1)  

#         box_est_featuresa = mask_global_max_pooling_2d(box_est_featuresa,mask)
#         box_est_featuresa = torch.cat([box_est_featuresa.view(-1, 640),one_hot_vec], -1)  
#         # box_est_features = torch.cat([box_est_features,one_hot_vec],1)
#         # print(box_est_features.shape)
#           # global max pooling
#         # print("2",box_est_features.shape)
#          # add one hot vec
#         # print("3",box_est_features.shape,rgb_features.shape)  

#         # box1 = self.box_est_head1(box_est_features)
#         box2 = self.box_est_head2(box_est_features)
#         box3 = self.box_est_head3(box_est_features)
#         box4 = self.box_est_head4(box_est_featuresa)
#         box  = result_selection_by_distance(stage1_center, box1, box2, box3,box4)
#         # print(box1.shape,box2.shape)
#         output_dict = parse_outputs(box, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, output_dict)
#         output_dict['center'] = output_dict['center_boxnet'] + stage1_center  # Bx3
#         return output_dict


# def result_selection_by_distance(center, box1, box2, box3, box4):
#     disntance = torch.zeros(center.shape[0], 1).cuda()
#     disntance[:, 0] = center[:, 2] # select batch dim, make mask shape (B, 1)
#     box = box1
#     #1
#     # box = torch.where(disntance < 15, box, box4)
#     box = torch.where(disntance < 15, box, box2)
#     # box = torch.where(disntance < 50, box, box3)
#     #2
#     box = torch.where(disntance < 40, box, box3)
#     box = torch.where(disntance < 60, box, box4)
    
#     return box


# if __name__ == '__main__':
#     import yaml
#     from lib.helpers.kitti_helper import Kitti_Config
#     dataset_config = Kitti_Config()
#     cfg = {'name': 'patchnet', 'init': 'xavier', 'threshold_offset': 0.5,
#            'patch_size': [32, 32], 'num_heading_bin': 12, 'num_size_cluster': 8,
#            'backbone': 'plainnet'}

#     input = torch.rand(2, 3, 64, 64)
#     one_hot = torch.Tensor(2, 3)

#     model = PatchNet(cfg,
#                      dataset_config.num_heading_bin,
#                      dataset_config.num_size_cluster,
#                      dataset_config.mean_size_arr)
#     output_dict = model(input, one_hot)
#     print (output_dict.keys())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.backbones.plainnet import PlainNet
from lib.backbones.senet import senet18_patchnet as senet
from lib.backbones.resnet import resnet18_patchnet as resnet
from lib.backbones.resnext import resnext_patchnet_1 as resnext
from lib.helpers.fpointnet_helper import parse_outputs
from lib.extensions.mask_global_pooling import mask_global_max_pooling_2d
from lib.extensions.mask_global_pooling import mask_global_avg_pooling_2d
from lib.helpers.misc_helper import init_weights
import os 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
#VIT
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head = 64,  dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    def forward(self,x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head,  dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        # x = x.squeeze(2)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
class CT(nn.Module):
    def __init__(self, num_patches, dim,depth, heads, mlp_dim, pool = 'mean', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.num_patches = num_patches
        
        patch_dim = channels * self.num_patches
        # print(patch_dim)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b n h w -> b n (h w)')
                        )
        self.to_patch_embedding2 = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.LayerNorm(dim)

    def forward(self, data, mask = None):
        x = data
        # print(x.shape)
        x = self.to_patch_embedding(x) # x.shape=[b,49,128]
        # print(x.shape)
        x = self.to_patch_embedding2(x)
        # print(x.shape)
       
        b, n, _ = x.shape # n = 49
        x += self.pos_embedding[:, :n] # x.shape=[b,50,128]
        x = self.dropout(x) 

        x = self.transformer(x) # x.shape=[b,50,128],mask=None
        # print("o",x.shape)
        x = x.mean(dim = 2) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        #使用自适应池化缩减map的大小，保持通道不变
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
        #map尺寸不变，缩减通道
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
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        # out = F.avg_pool2d(out, [32, 32])
        return out
class AM(nn.Module):
    def __init__(self):
        super(AM, self).__init__()
        # self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()
 
    def forward(self, x):
        # out = self.channel_attention(x) * x
        out = self.spatial_attention(x) * x
        # out = F.avg_pool2d(out, [32, 32])
        return out
class PatchNet(nn.Module):
    def __init__(self, cfg, num_heading_bin, num_size_cluster, mean_size_arr):
        super().__init__()
        self.cfg = cfg
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr

        # center estimation module
        self.center_reg_backbone = PlainNet(input_channels=3, layer_cfg=[128, 128, 256], kernal_size=1)
        self.center_reg_head = nn.Sequential(nn.Linear(259, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                             nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                             nn.Linear(128, 3))
        # box estiamtion module
        assert cfg['backbone'] in ['plainnet', 'resnet', 'resnext', 'senet']
        if cfg['backbone'] == 'plainnet':
            self.box_est_backbone = PlainNet(input_channels=3, layer_cfg=[128, 128, 256], kernal_size=3, padding=1)
            self.box_est_backbone2 = PlainNet(input_channels=256, layer_cfg=[512], kernal_size=3, padding=1)
        if cfg['backbone'] == 'resnet':
            self.box_est_backbone = resnet()
        if cfg['backbone'] == 'senet':
            self.box_est_backbone = senet()
        if cfg['backbone'] == 'resnext':
            self.box_est_backbone = resnext()

        self.box_est_head1 = nn.Sequential(nn.Linear(259, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                          nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                          nn.Linear(128, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))
        self.box_est_head2 = nn.Sequential(nn.Linear(643, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))
        
        self.box_est_head3 = nn.Sequential(nn.Linear(643, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))
        self.box_est_head4 = nn.Sequential(nn.Linear(643, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))

        self.img_backbone = resnet()
        self.attention2 = AM()
        self.attention = CBAM(channel=640)
        # self.transformer = CT(1024,dim=640,depth=3,heads=3,mlp_dim=1280,dropout=0.1,emb_dropout=0.1)
        init_weights(self, self.cfg['init'])


    def forward(self, patch1, one_hot_vec):
        output_dict = {}
        patch = patch1[:,:3,:,:]
        # get [h, w] of input patch  [for global pooling]
        _, _, h, w = patch.shape

        # mask generation
        rgb = patch1[:,3:,:,:]
        # import cv2 
        # print(rgb[0].permute(1,2,0).detach().cpu().numpy(),rgb[0].permute(1,2,0).detach().cpu().numpy().shape)
        # cv2.imwrite("test.png",rgb[0].permute(1,2,0).detach().cpu().numpy())
        depth_map = patch[:, 2, :, :]
        # print("0",patch.shape)
        threshold = depth_map.mean(-1).mean(-1) + self.cfg['threshold_offset']
        threshold = threshold.unsqueeze(-1).unsqueeze(-1).repeat(1, h, w)
        zeros, ones = torch.zeros_like(depth_map), torch.ones_like(depth_map)
        mask = torch.where(depth_map < threshold, ones, zeros)
        mask_xyz_mean = mask_global_avg_pooling_2d(patch, mask)
        patch = patch - mask_xyz_mean
        mask_xyz_mean = mask_xyz_mean.squeeze(-1).squeeze(-1)

        first_features = self.center_reg_backbone(patch)
        # box_est_features = self.box_est_backbone(first_features)
        
        # center regressor
        center_reg_features = mask_global_max_pooling_2d(first_features, mask)
        center_reg_features = torch.cat([center_reg_features.view(-1, 256), one_hot_vec], -1)  # add one hot vec
        center_tnet = self.center_reg_head(center_reg_features)
        stage1_center = center_tnet + mask_xyz_mean  # Bx3
        output_dict['stage1_center'] = stage1_center

        # first_features = self.attention2(first_features)
        # first_features = mask_global_max_pooling_2d(first_features, mask)
        # first_features = torch.cat([first_features.view(-1, 256), one_hot_vec], -1) 
        # box1 = self.box_est_head1(first_features)

        box_est_features1 = self.box_est_backbone(patch)
        box_est_features = self.box_est_backbone2(box_est_features1)
        # first_features = self.attention2(first_features)
        box_est_features1 = mask_global_max_pooling_2d(box_est_features1, mask)
        box_est_features1 = torch.cat([box_est_features1.view(-1, 256), one_hot_vec], -1) 
        box1 = self.box_est_head1(box_est_features1)
        # get patch in object coordinate
        patch = patch - center_tnet.unsqueeze(-1).unsqueeze(-1)

        # 3d box regressor
        # box_est_features = self.box_est_backbone(patch)

        rgb_features = self.img_backbone(rgb)
        # print(mask.shape)
        # rgb_features = mask_global_max_pooling_2d(rgb_features,mask)
        # rgb_features = torch.cat([rgb_features.view(-1, 256)], -1)
        # print("1",box_est_features.shape)
        # box_est_features = mask_global_max_pooling_2d(box_est_features, mask)
        # box_est_features = torch.cat([box_est_features.view(-1, 512)], -1)  
        box_est_features = torch.cat([box_est_features,rgb_features],1)
        box_est_featuresa = self.attention(box_est_features)
        # print(box_est_features.shape)
        # box_est_features = box_est_features.reshape((box_est_features.shape[0],box_est_features.shape[1],1,1))
        # print(box_est_features.shape)
        # box_est_features = self.transformer(box_est_features)
        box_est_features = mask_global_max_pooling_2d(box_est_features,mask)
        box_est_features = torch.cat([box_est_features.view(-1, 640),one_hot_vec], -1)  

        box_est_featuresa = mask_global_max_pooling_2d(box_est_featuresa,mask)
        box_est_featuresa = torch.cat([box_est_featuresa.view(-1, 640),one_hot_vec], -1)  
        # box_est_features = torch.cat([box_est_features,one_hot_vec],1)
        # print(box_est_features.shape)
          # global max pooling
        # print("2",box_est_features.shape)
         # add one hot vec
        # print("3",box_est_features.shape,rgb_features.shape)  

        # box1 = self.box_est_head1(box_est_features)
        box2 = self.box_est_head2(box_est_features)
        box3 = self.box_est_head3(box_est_features)
        box4 = self.box_est_head4(box_est_features)
        box  = result_selection_by_distance(stage1_center, box1, box2, box3,box4)
        # print(box1.shape,box2.shape)
        output_dict = parse_outputs(box, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, output_dict)
        output_dict['center'] = output_dict['center_boxnet'] + stage1_center  # Bx3
        return output_dict


def result_selection_by_distance(center, box1, box2, box3, box4):
    disntance = torch.zeros(center.shape[0], 1).cuda()
    disntance[:, 0] = center[:, 2] # select batch dim, make mask shape (B, 1)
    box = box1
    #1
    # box = torch.where(disntance < 15, box, box4)
    box = torch.where(disntance < 10, box, box2)
    # box = torch.where(disntance < 50, box, box3)
    #2
    box = torch.where(disntance < 30, box, box3)
    box = torch.where(disntance < 50, box, box4)
    
    return box


if __name__ == '__main__':
    import yaml
    from lib.helpers.kitti_helper import Kitti_Config
    dataset_config = Kitti_Config()
    cfg = {'name': 'patchnet', 'init': 'xavier', 'threshold_offset': 0.5,
           'patch_size': [32, 32], 'num_heading_bin': 12, 'num_size_cluster': 8,
           'backbone': 'plainnet'}

    input = torch.rand(2, 3, 64, 64)
    one_hot = torch.Tensor(2, 3)

    model = PatchNet(cfg,
                     dataset_config.num_heading_bin,
                     dataset_config.num_size_cluster,
                     dataset_config.mean_size_arr)
    output_dict = model(input, one_hot)
    print (output_dict.keys())

