import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from lib.backbones.plainnet import PlainNet
from lib.backbones.senet import senet18_patchnet as senet
from lib.backbones.resnet import resnet18_patchnet as resnet
from lib.backbones.resnext import resnext_patchnet_1 as resnext
from lib.helpers.fpointnet_helper import parse_outputs
from lib.extensions.mask_global_pooling import mask_global_max_pooling_2d,mask_global_max_pooling_2d2
from lib.extensions.mask_global_pooling import mask_global_avg_pooling_2d
from lib.helpers.misc_helper import init_weights
import os 
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Transformer
from .position_encoding import build_position_encoding,NerfPositionalEncoding
from .helpers import GenericMLP
from .common import iAFF,TransformerBlock
from .attention import *
class PlainNet(nn.Module):
    def __init__(self,
                 input_channels = 3,
                 layer_cfg = [128, 128, 256],
                 kernal_size = 1,
                 padding = 0,
                 batch_norm = True):
        super().__init__()
        self.input_channels = input_channels
        self.layer_cfg = layer_cfg
        self.kernal_size = kernal_size
        self.padding = padding
        self.batch_norm = batch_norm
        self.features = self.make_layers()


    def forward(self, patch):
        # print(patch.shape)
        return self.features(patch)


    def make_layers(self):
        layers = []
        input_channels = self.input_channels

        for output_channels in self.layer_cfg:
            if output_channels>0:
                layers += [nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                    kernel_size=self.kernal_size, padding=self.padding)]
                if self.batch_norm:
                    layers += [nn.BatchNorm2d(output_channels)]
                layers += [nn.ReLU(inplace=True)]
                input_channels = output_channels
            else:
                output_channels = -output_channels
                layers +=[TransformerBlock(output_channels,output_channels,2,1)]
                if self.batch_norm:
                    layers += [nn.BatchNorm2d(output_channels)]
                input_channels = output_channels
        return nn.Sequential(*layers)




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Conv, self).__init__()

        self.conv2 = torch.nn.Conv2d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm2d(outplanes)

    def forward(self, point_features, img_features):
        #print(point_features.shape, img_features.shape)

        fusion_features = torch.cat([point_features, img_features], dim=1)
        # fusion_features = fusion_features.permute(0,2,3,1)
        fusion_features = F.relu(self.bn1(self.conv2(fusion_features)))
        # print(fusion_features.shape)
        # fusion_features = fusion_features.permute(0,3,1,2)
        return fusion_features
class dot_attention(nn.Module):
    """ 点积注意力机制"""

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播
        :param q:
        :param k:
        :param v:
        :param scale:
        :param attn_mask:
        :return: 上下文张量和attention张量。
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        # 是否设置缩放
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)     # 给需要mask的地方设置一个负无穷。
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和v做点积。
        context = torch.bmm(attention, v)
        return context, attention
class PatchNet(nn.Module):
    def __init__(self, cfg, num_heading_bin, num_size_cluster, mean_size_arr):
        super().__init__()
        self.cfg = cfg
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        # print(cfg)
        # center estimation module
        self.center_reg_backbone = PlainNet(input_channels=3, layer_cfg=[128, 128, 256], kernal_size=1)
        self.center_reg_head = nn.Sequential(nn.Linear(259, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                             nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                             nn.Linear(128, 3))
        # self.classweight1 = nn.Linear(3, 128)
        self.classweight2 = nn.Linear(3, 512)
        self.attn1 = dot_attention()
        self.attn2 = dot_attention()
        # box estiamtion module
        assert cfg['backbone'] in ['plainnet', 'resnet', 'resnext', 'senet']
        if cfg['backbone'] == 'plainnet':
            if self.cfg['fusion']=="afterward" or self.cfg['fusion']=="None":
                self.box_est_backbone = PlainNet(input_channels=3, layer_cfg=[128, 128, 256, 512], kernal_size=3, padding=1)
            elif self.cfg['fusion']=="deep":
                self.box_est_backbone = nn.ModuleList([nn.Sequential(PlainNet(input_channels=3,layer_cfg=[128,128], kernal_size=3, padding=1)),
                                                      PlainNet(input_channels=128,layer_cfg=[256],kernal_size=3,padding=1),
                                                      PlainNet(input_channels=256,layer_cfg=[512],kernal_size=3,padding=1)])
            elif self.cfg['fusion']=="deep2":
                self.box_est_backbone = nn.ModuleList([nn.Sequential(PlainNet(input_channels=3,layer_cfg=[128,128], kernal_size=3, padding=1)),
                                                      PlainNet(input_channels=160,layer_cfg=[288],kernal_size=3,padding=1),
                                                      PlainNet(input_channels=352,layer_cfg=[608],kernal_size=3,padding=1)])
                # self.box_est_bcbone1 = PlainNet(input_channels=3,layer_cfg=[128,128], kernal_size=3, padding=1)
                # self.box_est_bcbone2 = PlainNet(input_channels=128,layer_cfg=[256],kernal_size=3,padding=1)
                # self.box_est_bcbone3 = PlainNet(input_channels=256,layer_cfg=[512],kernal_size=3,padding=1)
            elif self.cfg['fusion']=="forward":
                self.box_est_backbone = PlainNet(input_channels=6, layer_cfg=[128, 128, 256, 512], kernal_size=3, padding=1)
        if cfg['backbone'] == 'resnet':
            self.box_est_backbone = resnet()
        if cfg['backbone'] == 'senet':
            self.box_est_backbone = senet()
        if cfg['backbone'] == 'resnext':
            self.box_est_backbone = resnext()


        self.featuredim = 512

        if self.cfg['fusion']=="afterward":
            self.img_backbone = resnet()
            self.featuredim = 512 + self.cfg['imgchannel']
        elif self.cfg['fusion']=="deep":
            self.img_backbone = nn.ModuleList([BasicBlock(3,32,1),
                                 BasicBlock(32, 64, 1), BasicBlock(64, 128, 1)])
            self.fusion_conv = nn.ModuleList([Fusion_Conv(160,128),Fusion_Conv(320,256),Fusion_Conv(640,512)])
        elif self.cfg['fusion'] == "deep2":
            self.img_backbone = nn.ModuleList([BasicBlock(3, 32, 1),
                                               BasicBlock(32, 64, 1), BasicBlock(64, 128, 1)])
            self.featuredim = 736
        self.box_est_head1 = nn.Sequential(nn.Linear(self.featuredim+3, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))
        self.box_est_head2 = nn.Sequential(nn.Linear(self.featuredim+3, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))
        self.box_est_head3 = nn.Sequential(nn.Linear(self.featuredim+3, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))
        if self.cfg['attention']:
            if self.cfg['attentiontype']=="plus" or self.cfg['attentiontype']=='class':
                self.attn = AdditiveAttention(512,128,256,0.5)
                # self.attn = AdditiveAttention(1024, 1024, 256, 0.5)
            elif self.cfg['attentiontype']=="plusweight":
                self.attn = AdditiveAttention(1024, 1024, 256, 0.5)
            elif self.cfg['attentiontype']=="dot":
                self.attn = DotProductAttention(0.5)
            elif self.cfg['attentiontype']=="CBAM":
                self.attn = CBAM(640)
            elif self.cfg['attentiontype']=="iAFF":
                self.attn = iAFF(512,2)
            elif self.cfg['attentiontype']=="MultiheadAttn":
                self.attn = MultiHeadAttention()
            elif self.cfg['attentiontype']=="co":
                self.attn = ParallelCoAttentionNetwork(512,128,512,src_length_masking=False)
        init_weights(self, self.cfg['init'])

    def forward(self, patch1, one_hot_vec,save=False,epoch=None):
        output_dict = {}
        patch = patch1[:,:3,:,:]
        # get [h, w] of input patch  [for global pooling]
        _, _, h, w = patch.shape

        # mask generation
        rgb = patch1[:,3:,:,:]

        depth_map = patch[:, 2, :, :]
        threshold = depth_map.mean(-1).mean(-1) + self.cfg['threshold_offset']
        threshold = threshold.unsqueeze(-1).unsqueeze(-1).repeat(1, h, w)
        zeros, ones = torch.zeros_like(depth_map), torch.ones_like(depth_map)
        mask = torch.where(depth_map < threshold, ones, zeros)
        mask_xyz_mean = mask_global_avg_pooling_2d(patch, mask)
        
        patch = patch - mask_xyz_mean
        mask_xyz_mean = mask_xyz_mean.squeeze(-1).squeeze(-1)

        # center regressor
        first_features = self.center_reg_backbone(patch)
        center_reg_features = mask_global_max_pooling_2d(first_features, mask)
        center_reg_features = torch.cat([center_reg_features.view(-1, 256), one_hot_vec], -1)  # add one hot vec

        center_tnet = self.center_reg_head(center_reg_features)
        stage1_center = center_tnet + mask_xyz_mean  # Bx3
        output_dict['stage1_center'] = stage1_center

        # get patch in object coordinate
        patch = patch - center_tnet.unsqueeze(-1).unsqueeze(-1)

        if self.cfg['fusion']=="afterward":
            rgb_features = self.img_backbone(rgb)
            box_est_features = self.box_est_backbone(patch)
            if not self.cfg['attention']:
                box_est_features = torch.cat([box_est_features,rgb_features],dim=1)
                box_est_features = mask_global_max_pooling_2d(box_est_features, mask)
                # print(box_est_features.shape)
            else:
                if self.cfg['attentiontype']=="class":
                    # box_est_features = box_est_features.flatten(2)
                    # rgb_features = rgb_features.flatten(2)
                    # print(rgb_features.shape,box_est_features.shape)

                    # box_est_features = self.attn(rgb_features, box_est_features, one_hot_vec)
                    # print(box_est_features.shape)
                    # rgb_features = rgb_features.permute(0,2,1)
                    # box_est_features = box_est_features.permute(0,2,1)
                    # atten = atten.permute(0, 2, 1).reshape(atten.shape[0], 2, 32, 32)

                    # box_est_features = box_est_features.reshape(box_est_features.shape[0], box_est_features.shape[1],
                    #                                             32, 32)
                    # box_est_features = box_est_features*atten[:,0,:,:].unsqueeze(1)
                    # box_est_features = box_est_features.squeeze(1)
                    # print(box_est_features.shape,mask.shape)
                    # box_est_features = mask_global_max_pooling_2d(box_est_features, mask)
                    # rgb_features = rgb_features.reshape(rgb_features.shape[0], rgb_features.shape[1],
                    #                                     32, 32)
                    # rgb_features = rgb_features * atten[:, 1, :, :].unsqueeze(1)
                    #
                    # rgb_features = mask_global_max_pooling_2d(rgb_features, mask)
                    # print(atten[0,0,:,:],"***",atten[0,1,:,:])
                    # if save:
                    # print(atten.shape)
                    # for k in range(atten.shape[0]):
                    #     pw = torch.sum(atten[k, 0, :, :]).detach().cpu().numpy()
                    #     iw = torch.sum(atten[k, 1, :, :]).detach().cpu().numpy()
                    #     f = open("weight_all.csv", "a", encoding='utf-8')
                    #     csv_writer = csv.writer(f)
                    #     csv_writer.writerow([str(float(pw)), str(float(iw))])
                    #     f.close()
                    # with open("weight_car_val.txt","a")as f:
                    # f.write(str(epoch)+" "+str(float(pw))+" "+str(float(iw))+"\n")
                    # f.write(str(float(pw))+" "+str(float(iw))+"\n")
                    # box_est_features = box_est_features.view(-1,box_est_features.shape[1])
                    # rgb_features = rgb_features.view(-1,rgb_features.shape[1])
                    # rgb_features = self.attn(rgb_features,box_est_features,rgb_features)
                    # rgb_features = rgb_features.squeeze(1)
                    # box_est_features = torch.cat([box_est_features, rgb_features], dim=1)
                    # one_hot_vec1 = self.classweight1(one_hot_vec)
                    one_hot_vec2 = self.classweight2(one_hot_vec)
                    # one_hot_vec1 = one_hot_vec1.view(-1, 128, 1).repeat(1, 1, 1024)
                    one_hot_vec2 = one_hot_vec2.view(-1, 512, 1).repeat(1, 1, 1024)
                    context1, attention1 = self.attn1(one_hot_vec2, rgb_features.flatten(2), rgb_features.flatten(2))
                    context2, attention2 = self.attn2(one_hot_vec2, box_est_features.flatten(2),
                                                      box_est_features.flatten(2))
                    box_est_features = torch.cat([context2.reshape(-1, 512, 32, 32), context1.reshape(-1, 512, 32, 32)], dim=1)
                    box_est_features = mask_global_max_pooling_2d(box_est_features, mask)
                if self.cfg['attentiontype']=="plus" or self.cfg['attentiontype'] =="dot":
                    # print(box_est_features.shape,rgb_features.shape)
                    box_est_features = box_est_features.flatten(2)
                    rgb_features = rgb_features.flatten(2)
                    # print(rgb_features.shape,box_est_features.shape)
                    atten = self.attn(rgb_features, box_est_features, one_hot_vec)
                    # print(box_est_features.shape)
                    # rgb_features = rgb_features.permute(0,2,1)

                    atten = atten.permute(0,2,1).reshape(atten.shape[0],2,32,32)

                    box_est_features = box_est_features.reshape(box_est_features.shape[0],box_est_features.shape[1],32,32)
                    # box_est_features = box_est_features*atten[:,0,:,:].unsqueeze(1)
                    box_est_features = box_est_features.squeeze(1)
                    # print(box_est_features.shape,mask.shape)
                    box_est_features = mask_global_max_pooling_2d(box_est_features, mask)
                    rgb_features = rgb_features.reshape(rgb_features.shape[0], rgb_features.shape[1],
                                                                32, 32)
                    rgb_features = rgb_features*atten[:,1,:,:].unsqueeze(1)

                    rgb_features = mask_global_max_pooling_2d(rgb_features, mask)
                    # print(atten[0,0,:,:],"***",atten[0,1,:,:])
                    # if save:
                    # print(atten.shape)
                    # for k in range(atten.shape[0]):
                    #     pw = torch.sum(atten[k, 0, :, :]).detach().cpu().numpy()
                    #     iw = torch.sum(atten[k, 1, :, :]).detach().cpu().numpy()
                    #     f = open("weight_all.csv", "a", encoding='utf-8')
                    #     csv_writer = csv.writer(f)
                    #     csv_writer.writerow([str(float(pw)), str(float(iw))])
                    #     f.close()
                    # with open("weight_car_val.txt","a")as f:
                        # f.write(str(epoch)+" "+str(float(pw))+" "+str(float(iw))+"\n")
                        # f.write(str(float(pw))+" "+str(float(iw))+"\n")
                    # box_est_features = box_est_features.view(-1,box_est_features.shape[1])
                    # rgb_features = rgb_features.view(-1,rgb_features.shape[1])
                    # rgb_features = self.attn(rgb_features,box_est_features,rgb_features)
                    # rgb_features = rgb_features.squeeze(1)

                    box_est_features = torch.cat([box_est_features, rgb_features], dim=1)
                elif self.cfg['attentiontype'] == "co":
                    box_est_features = box_est_features.flatten(2)
                    rgb_features = rgb_features.flatten(2)
                    a_v, a_q, box_est_features, rgb_features = self.attn(box_est_features,rgb_features,rgb_features.shape[0])
                    # print(box_est_features.shape,rgb_features.shape)
                    box_est_features = torch.cat([box_est_features,rgb_features],dim=1)
                elif self.cfg['attentiontype']=="MultiheadAttn":
                    box_est_features = mask_global_max_pooling_2d(box_est_features, mask)
                    rgb_features = mask_global_max_pooling_2d(rgb_features, mask)
                    box_est_features = box_est_features.view(-1,box_est_features.shape[1])
                    rgb_features = rgb_features.view(-1,rgb_features.shape[1])
                    rgb_features = rgb_features.unsqueeze(1)
                    box_est_features2 = box_est_features.unsqueeze(1)
                    rgb_features,attention = self.attn(box_est_features2,rgb_features,rgb_features)
                    # print(rgb_features.shape)
                    rgb_features = rgb_features.squeeze(1)
                    box_est_features = torch.cat([box_est_features,rgb_features],dim=1)
                elif self.cfg['attentiontype']=="CBAM":
                    box_est_features = torch.cat([box_est_features, rgb_features], dim=1)
                    # print(box_est_features.shape)
                    box_est_features = self.attn(box_est_features)
                    # print(box_est_features.shape)
                    box_est_features = mask_global_max_pooling_2d(box_est_features, mask)
                    # print(box_est_features.shape)
                    # box_est_features = box_est_features.view(-1, box_est_features.shape[1])
        elif self.cfg['fusion']=="deep":
            rgb_features = rgb
            box_est_features = patch
            for i in range(len(self.img_backbone)):
                rgb_features = self.img_backbone[i](rgb_features)
                box_est_features = self.box_est_backbone[i](box_est_features)
                box_est_features = self.fusion_conv[i](box_est_features,rgb_features)
            box_est_features = mask_global_max_pooling_2d(box_est_features, mask)
        elif self.cfg['fusion']=="deep2":
            rgb_features = rgb
            box_est_features = patch
            for i in range(len(self.img_backbone)):
                rgb_features = self.img_backbone[i](rgb_features)
                box_est_features = self.box_est_backbone[i](box_est_features)
                box_est_features = torch.cat([box_est_features,rgb_features],dim=1)
            box_est_features = mask_global_max_pooling_2d(box_est_features, mask)
        elif self.cfg['fusion']=="forward":
            box_est_features = self.box_est_backbone(patch1)
            box_est_features = mask_global_max_pooling_2d(box_est_features, mask)
        elif self.cfg['fusion']=="None":
            box_est_features = self.box_est_backbone(patch)

            box_est_features = mask_global_max_pooling_2d(box_est_features, mask)

        box_est_features = torch.cat([box_est_features.view(-1,self.featuredim),one_hot_vec],1)

        box1 = self.box_est_head1(box_est_features)
        box2 = self.box_est_head2(box_est_features)
        box3 = self.box_est_head3(box_est_features)
        box  = result_selection_by_distance(stage1_center, box1, box2, box3)
        output_dict = parse_outputs(box, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, output_dict)
        output_dict['center'] = output_dict['center_boxnet'] + stage1_center  # Bx3
        return output_dict


def result_selection_by_distance(center, box1, box2, box3):
    disntance = torch.zeros(center.shape[0], 1).cuda()
    disntance[:, 0] = center[:, 2] # select batch dim, make mask shape (B, 1)
    box = box1
    box = torch.where(disntance < 30, box, box2)
    box = torch.where(disntance < 50, box, box3)
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

