'''
F-PointNet, http://openaccess.thecvf.com/content_cvpr_2018/html/Qi_Frustum_PointNets_for_CVPR_2018_paper.html
'''
import csv
import torch
import torch.nn as nn
from torch.nn.functional import grid_sample
from lib.backbones.pointnet import PointNet
from lib.backbones.pointnet import PointNet_SEG
from lib.helpers.fpointnet_helper import point_cloud_masking
from lib.helpers.fpointnet_helper import parse_outputs
from lib.helpers.misc_helper import init_weights
from lib.backbones.resnet import resnet18_patchnet as resnet
from .attention2 import *
import cv2
import numpy as np
import torch.nn.functional as F
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

class FPointNet(nn.Module):
    def __init__(self, cfg, num_heading_bin, num_size_cluster, mean_size_arr):
        super().__init__()
        self.cfg = cfg
        self.mean_size_arr = mean_size_arr  # fixed anchor
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.point_seg_backbone = PointNet_SEG(num_points=self.cfg['num_frustum_point'],
                                               input_channels=cfg['input_channel'])
        if self.cfg['fusion']=="afterward":
            self.point_seg_head = nn.Sequential(nn.Dropout(0.5),
                                            nn.Conv1d(259, 2, 1))
        else:
            self.point_seg_head = nn.Sequential(nn.Dropout(0.5),
                                                nn.Conv1d(128, 2, 1))

        self.center_reg_backbone = PointNet(num_points=self.cfg['num_object_points'],
                                            input_channels=3,
                                            layer_cfg=[128, 128, 256],
                                            batch_norm=True)
        # self.center_reg_backbone1 = PointNet(num_points=self.cfg['num_object_points'],
        #                                     input_channels=3,
        #                                     layer_cfg=[128, 128, 256],
        #                                     batch_norm=True)
        self.center_reg_head = nn.Sequential(nn.Linear(259, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                             nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                             nn.Linear(128, 3))
        self.box_est_backbone = PointNet(num_points=self.cfg['num_object_points'],
                                         input_channels=3,
                                         layer_cfg=[128, 128, 256, 512],
                                         batch_norm=True)
        # self.box_est_backbone1 = PointNet(num_points=self.cfg['num_object_points2'],
        #                                  input_channels=3,
        #                                  layer_cfg=[128, 128, 256, 512],
        #                                  batch_norm=True)
        self.box_est_head = nn.Sequential(nn.Linear(515, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                          nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                          nn.Linear(256, 3 + self.num_heading_bin*2 + self.num_size_cluster*4))

        init_weights(self, self.cfg['init'])
        self.img_backbone = resnet()
        self.classweight = nn.Linear(3,128)
        self.attn1 = dot_attention()
        self.attn2 = dot_attention()
        # self.attn = SE(256,256)
        # self.attn = AdditiveAttention(128,128,128,0.5)
        # self.attnco = ParallelCoAttentionNetwork(128,128,False)
    def forward(self, point_cloud, one_hot_vec,box2d=None,localimage=None,save=False,epoch=None):
        output_dict = {}
        point_cloud = point_cloud.float()

        # print(point_cloud.shape,point_cloud[0,:,:10],box2d.shape,localimage.shape)
        # print(point_cloud)
        # print(localimage.shape)
        # localimage = localimage.permute(0,3,1,2)

        # localimage = F.interpolate(localimage.cuda().float(), [100,100], mode='bilinear', align_corners=True).squeeze(0).detach().cpu().numpy()


        box2d1 = box2d.unsqueeze(0).permute(1,0,2).repeat(1,1024,1)
        # xyor = point_cloud[:,7:9,:]
        xyor = point_cloud[:, 4:6, :]
        # print(xyor)
        xyor = xyor.permute(0,2,1)
        # print(xyor.shape,box2d.shape)
        xyor[:,:,0] = xyor[:,:,0]-box2d1[:,:,0]
        xyor[:,:,1] = xyor[:,:,1]-box2d1[:,:,1]
        xy = xyor
        # size_range = [box2d[], 500.0]
        size_range = [box2d1[:,:,2]-box2d1[:,:,0],box2d1[:,:,3]-box2d1[:,:,1]]
        xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
        xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0  # = xy / (size_range - 1.) * 2 - 1.
        # l_xy_cor = [xy]
        # img = [imgfeatures]
        xyt = xy.unsqueeze(1)
        # localimage = F.interpolate(localimage.cuda().float(), [32,32], mode='bilinear', align_corners=True).squeeze(0)
        ###
        imgfeatures = self.img_backbone(localimage.cuda().float())
        # print(localimage.shape)
        # cv2.imwrite("image.png", localimage[0].permute(1,2,0).detach().cpu().numpy())
        # print(localimage[0].shape)
        pointsrgb = grid_sample(localimage.float(), xyt, mode='nearest')
        # print(pointsrgb.shape,point_cloud[:,:3,:].shape)
        pointsrgb = pointsrgb.squeeze(2)
        pointsrgb = torch.cat([point_cloud[:,:3,:],pointsrgb],dim=1)
        # np.savetxt("pointsrgb.txt",pointsrgb[0].permute(1,0).detach().cpu().numpy())
        # print(localimage[0].shape)
        # cv2.imwrite("test.png",localimage[0].permute(1,2,0).detach().cpu().numpy())
        # background points segmentation
        point_cloud1 = point_cloud[:,:3,:]
        # point_cloud = torch.cat([point_cloud[:,:3,:],pointsrgb],dim=1)
        # print(xyt.shape)
        imgfeatures1 = grid_sample(imgfeatures, xyt, mode='nearest')
        # # print(imgfeatures.shape)
        imgfeatures1 = imgfeatures1.squeeze(2)

        seg_features = self.point_seg_backbone(point_cloud1,one_hot_vec)
        # print(seg_features.shape,imgfeatures.shape)

        # print(imgfeatures1.shape)
        # print(seg_features.shape,imgfeatures.shape)

        # atten = self.attn(seg_features,imgfeatures1)
        # attns,attni,seg_features,imgfeatures=self.attnco(seg_features, imgfeatures,32)
        # print(seg_features.shape,imgfeatures.shape)
        # print(atten.shape)
        # atten = atten.permute(0, 2, 1)
        # #
        # seg_features = seg_features * atten[:, 0, :].unsqueeze(1)
        # imgfeatures = seg_features * atten[:, 1, :].unsqueeze(1)
        # save = True
        # if save:
        #     for k in range(atten.shape[0]):
        #         pw = torch.sum(atten[k, 0, :]).detach().cpu().numpy()
        #         iw = torch.sum(atten[k, 1, :]).detach().cpu().numpy()
        #         f = open("weight_all_fpoint_all.csv", "a", encoding='utf-8')
        #         csv_writer = csv.writer(f)
        #         csv_writer.writerow([str(float(pw)), str(float(iw))])
        #         f.close()
        # print(torch.sum(atten[0, 0, :])
        #                 ,torch.sum(atten[0, 1, :]))
        # seg_features = seg_features.permute(0,2,1)
        # imgfeatures = imgfeatures.permute(0,2,1)
        # print(seg_features.shape,imgfeatures.shape)
        # seg_features = torch.cat([seg_features],dim=1)
        # fusion_feature = self.attn(torch.cat([seg_features,imgfeatures1],dim=1))
        # print(seg_features.shape,imgfeatures1.shape)
        # one_hot_vec1 = self.classweight(one_hot_vec)
        # one_hot_vec1 = one_hot_vec1.view(-1, 128, 1).repeat(1, 1, 1024)
        one_hot_vec1 = one_hot_vec.view(-1, 3, 1).repeat(1, 1, 1024)
        # context1, attention1 = self.attn1(imgfeatures1,seg_features,seg_features)
        #cls1
        # context1, attention1 = self.attn1(seg_features, one_hot_vec1, seg_features)
        # context2, attention2 = self.attn2(imgfeatures1, one_hot_vec1, imgfeatures1)
        #cls2
        # context1, attention1 = self.attn1(one_hot_vec1, seg_features, seg_features)
        # context2, attention2 = self.attn2(one_hot_vec1, imgfeatures1, imgfeatures1)
        # fusion_feature = torch.cat([context1,context2],dim=1)
        fusion_feature = torch.cat([seg_features,imgfeatures1,one_hot_vec1],dim=1)
        seg_logits = self.point_seg_head(fusion_feature).transpose(1, 2).contiguous()    # B*C*N -> B*N*C
        output_dict['mask_logits'] = seg_logits
        # print(point_cloud.shape)
        # background points masking
        # select masked points and translate to masked points' centroid
        point_cloud = point_cloud.transpose(1, 2).contiguous()  # B*C*N -> B*N*C, meet the shape requirements of funcs
        object_point_cloud_xyz, mask_xyz_mean, output_dict = \
            point_cloud_masking(point_cloud, seg_logits, self.cfg['num_object_points'], output_dict,xyz_only =True)
        # print(seg_logits.shape)

        # T-Net and coordinate translation
        # print(object_point_cloud_xyz.shape)
        # object_point_cloud_xyco = object_point_cloud_xyz[:, :, 4:]
        object_point_cloud_xyz = object_point_cloud_xyz[:,:,:3]
        # print(object_point_cloud_xyz.shape)

        # box2d = box2d.unsqueeze(0).permute(1, 0, 2).repeat(1, 512, 1)
        # xyor = object_point_cloud_xyco
        # # xyor = xyor.permute(0,2,1)
        # # print(xyor.shape,box2d.shape)
        # # xyor = xyor.permute(0,2,1)
        # xyor[:,:,0] = xyor[:,:,0]-box2d[:,:,0]
        # xyor[:,:,1] = xyor[:,:,1]-box2d[:,:,1]
        # xy = xyor
        # # size_range = [box2d[], 500.0]
        # size_range = [box2d[:,:,2]-box2d[:,:,0],box2d[:,:,3]-box2d[:,:,1]]
        # xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
        # xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0  # = xy / (size_range - 1.) * 2 - 1.
        # # l_xy_cor = [xy]
        # # img = [imgfeatures]
        # xyt = xy.unsqueeze(1)
        # # print(xyt.shape,imgfeatures.shpae)
        # imgfeatures = grid_sample(imgfeatures, xyt, mode='nearest')
        # # print(imgfeatures.shape)
        # imgfeatures = imgfeatures.squeeze(2)

        object_point_cloud_xyz = object_point_cloud_xyz.transpose(1, 2).contiguous()  # B*N*C -> B*C*N
        center_features = self.center_reg_backbone(object_point_cloud_xyz).view(-1, 256)  #  B*C*1 -> B*C
        # print(center_features.shape,"113")
        center_features = torch.cat([center_features, one_hot_vec], 1)
        center_tnet = self.center_reg_head(center_features)
        stage1_center = center_tnet + mask_xyz_mean  # Bx3
        output_dict['stage1_center'] = stage1_center

        # Get object point cloud in object coordinate
        object_point_cloud_xyz_new = object_point_cloud_xyz - center_tnet.view(-1, 3, 1)
        # print(object_point_cloud_xyz_new.shape)
        # Amodel Box Estimation PointNet
        # print(object_point_cloud_xyz_new.shape, "**", imgfeatures.shape)
        # object_point_cloud_xyz_new = torch.cat([object_point_cloud_xyz_new,imgfeatures],dim=1)
        box_features = self.box_est_backbone(object_point_cloud_xyz_new).view(-1, 512)  # B*C*1 -> B*C
        # imgfeatures = F.max_pool1d(imgfeatures, 512).view(-1,128)

        box_features = torch.cat([box_features,one_hot_vec], 1)
        box_results = self.box_est_head(box_features)
        output_dict = parse_outputs(box_results, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, output_dict)
        output_dict['center'] = output_dict['center_boxnet'] + stage1_center  # Bx3

        return output_dict




if __name__ == '__main__':
    # only for debug
    import yaml
    import numpy as np
    cfg_file = '../../experiments/fpointnet/config_fpointnet.yaml'
    cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)['model']
    fpointnet = FPointNet(cfg, 12, 8, np.zeros((8, 3), dtype=np.float32))

    points = torch.Tensor(2, 4, 1024)
    one_hot_vec = torch.Tensor(2, 3)

    outputs = fpointnet(points, one_hot_vec)
    for key in outputs:
        print((key, outputs[key].shape))
    print(len(outputs.keys()))