# import torch
# import torch.nn as nn

# from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
# from ...ops.roipoint_pool3d import roipoint_pool3d_utils
# from ...utils import common_utils
# from .roi_head_template import RoIHeadTemplate
# from ..model_utils.ctrans import build_transformer
# import torch.nn.functional as F
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
# def transformer_neighbors(x, feature, k=20, idx=None):
#     '''
#         input: x, [B,3,N]
#                feature, [B,C,N]
#         output: neighbor_x, [B,6,N,K]
#                 neighbor_feat, [B,2C,N,k]
#     '''
#     batch_size = x.size(0)
#     num_points = x.size(2)
#     x = x.view(batch_size, -1, num_points)
#     if idx is None:
#         idx = knn(x, k=k)  # (batch_size, num_points, k)
#     device = torch.device('cuda')

#     idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
#     idx_base = idx_base.type(torch.cuda.LongTensor)
#     idx = idx.type(torch.cuda.LongTensor)
#     idx = idx + idx_base
#     idx = idx.view(-1)

#     _, num_dims, _ = x.size()

#     x = x.transpose(2,
#                     1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
#     neighbor_x = x.view(batch_size * num_points, -1)[idx, :]
#     neighbor_x = neighbor_x.view(batch_size, num_points, k, num_dims)
#     x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

#     position_vector = (x - neighbor_x).permute(0, 3, 1, 2).contiguous()  # B,3,N,k

#     _, num_dims, _ = feature.size()

#     feature = feature.transpose(2,
#                                 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
#     neighbor_feat = feature.view(batch_size * num_points, -1)[idx, :]
#     neighbor_feat = neighbor_feat.view(batch_size, num_points, k, num_dims)
#     neighbor_feat = neighbor_feat.permute(0, 3, 1, 2).contiguous()  # B,C,N,k

#     return position_vector, neighbor_feat
# class Point_Transformer(nn.Module):
#     def __init__(self, input_features_dim):
#         super(Point_Transformer, self).__init__()

#         self.conv_theta1 = nn.Conv2d(3, input_features_dim, 1)
#         self.conv_theta2 = nn.Conv2d(input_features_dim, input_features_dim, 1)
#         self.bn_conv_theta = nn.BatchNorm2d(input_features_dim)

#         self.conv_phi = nn.Conv2d(input_features_dim, input_features_dim, 1)
#         self.conv_psi = nn.Conv2d(input_features_dim, input_features_dim, 1)
#         self.conv_alpha = nn.Conv2d(input_features_dim, input_features_dim, 1)

#         self.conv_gamma1 = nn.Conv2d(input_features_dim, input_features_dim, 1)
#         self.conv_gamma2 = nn.Conv2d(input_features_dim, input_features_dim, 1)
#         self.bn_conv_gamma = nn.BatchNorm2d(input_features_dim)

#     def forward(self, xyz, features, k):

#         position_vector, x_j = transformer_neighbors(xyz, features, k=k)

#         delta = F.relu(self.bn_conv_theta(self.conv_theta2(self.conv_theta1(position_vector)))) # B,C,N,k
#         # corrections for x_i
#         x_i = torch.unsqueeze(features, dim=-1).repeat(1, 1, 1, k) # B,C,N,k

#         linear_x_i = self.conv_phi(x_i) # B,C,N,k

#         linear_x_j = self.conv_psi(x_j) # B,C,N,k

#         relation_x = linear_x_i - linear_x_j + delta # B,C,N,k
#         relation_x = F.relu(self.bn_conv_gamma(self.conv_gamma2(self.conv_gamma1(relation_x)))) # B,C,N,k

#         weights = F.softmax(relation_x, dim=-1) # B,C,N,k
#         features = self.conv_alpha(x_j) + delta # B,C,N,k

#         f_out = weights * features # B,C,N,k
#         f_out = torch.sum(f_out, dim=-1) # B,C,N

#         return f_out

# class PointRCNNHead(RoIHeadTemplate):
#     def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
#         super().__init__(num_class=num_class, model_cfg=model_cfg)
#         self.model_cfg = model_cfg
#         use_bn = self.model_cfg.USE_BN
#         self.SA_modules = nn.ModuleList()
#         channel_in = input_channels

#         self.num_prefix_channels = 3 + 2  # xyz + point_scores + point_depth
#         xyz_mlps = [self.num_prefix_channels] + self.model_cfg.XYZ_UP_LAYER
#         # shared_mlps = []
#         # for k in range(len(xyz_mlps) - 1):
#         #     shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))
#         #     if use_bn:
#         #         shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
#         #     shared_mlps.append(nn.ReLU())
#         # self.xyz_up_layer = nn.Sequential(*shared_mlps)

#         # c_out = self.model_cfg.XYZ_UP_LAYER[-1]
#         # self.merge_down_layer = nn.Sequential(
#         #     nn.Conv2d(c_out * 2, c_out, kernel_size=1, bias=not use_bn),
#         #     *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
#         # )

#         num_queries = model_cfg.Transformer.num_queries
#         hidden_dim = model_cfg.Transformer.hidden_dim
#         self.num_points = model_cfg.Transformer.num_points

#         # self.class_embed = nn.Linear(hidden_dim, 1)
#         # self.bbox_embed = MLP(hidden_dim, hidden_dim, self.box_coder.code_size * self.num_class, 4)
#         self.query_embed = nn.Embedding(num_queries, hidden_dim)
#         self.transformer = build_transformer(model_cfg.Transformer)

#         # for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
#         #     mlps = [channel_in] + self.model_cfg.SA_CONFIG.MLPS[k]

#         #     npoint = self.model_cfg.SA_CONFIG.NPOINTS[k] if self.model_cfg.SA_CONFIG.NPOINTS[k] != -1 else None
#         #     self.SA_modules.append(
#         #         pointnet2_modules.PointnetSAModule(
#         #             npoint=npoint,
#         #             radius=self.model_cfg.SA_CONFIG.RADIUS[k],
#         #             nsample=self.model_cfg.SA_CONFIG.NSAMPLE[k],
#         #             mlp=mlps,
#         #             use_xyz=True,
#         #             bn=use_bn
#         #         )
#         #     )
#         #     channel_in = mlps[-1]
#         self.up_dimension = MLP(input_dim = 128, hidden_dim = 256, output_dim = 512, num_layers = 2)
#         self.cls_layers = self.make_fc_layers(
#             input_channels=channel_in, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
#         )
#         self.reg_layers = self.make_fc_layers(
#             input_channels=channel_in,
#             output_channels=self.box_coder.code_size * self.num_class,
#             fc_list=self.model_cfg.REG_FC
#         )

#         self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
#             num_sampled_points=self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS,
#             pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH
#         )
#         self.init_weights(weight_init='xavier')

#     def init_weights(self, weight_init='xavier'):
#         if weight_init == 'kaiming':
#             init_func = nn.init.kaiming_normal_
#         elif weight_init == 'xavier':
#             init_func = nn.init.xavier_normal_
#         elif weight_init == 'normal':
#             init_func = nn.init.normal_
#         else:
#             raise NotImplementedError

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
#                 if weight_init == 'normal':
#                     init_func(m.weight, mean=0, std=0.001)
#                 else:
#                     init_func(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#         nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

#     def roipool3d_gpu(self, batch_dict):
#         """
#         Args:
#             batch_dict:
#                 batch_size:
#                 rois: (B, num_rois, 7 + C)
#                 point_coords: (num_points, 4)  [bs_idx, x, y, z]
#                 point_features: (num_points, C)
#                 point_cls_scores: (N1 + N2 + N3 + ..., 1)
#                 point_part_offset: (N1 + N2 + N3 + ..., 3)
#         Returns:

#         """
#         batch_size = batch_dict['batch_size']
#         batch_idx = batch_dict['point_coords'][:, 0]
#         point_coords = batch_dict['point_coords'][:, 1:4]
#         point_features = batch_dict['point_features']
#         rois = batch_dict['rois']  # (B, num_rois, 7 + C)
#         batch_cnt = point_coords.new_zeros(batch_size).int()
#         for bs_idx in range(batch_size):
#             batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

#         assert batch_cnt.min() == batch_cnt.max()

#         point_scores = batch_dict['point_cls_scores'].detach()
#         point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER - 0.5
#         point_features_list = [point_scores[:, None], point_depths[:, None], point_features]
#         point_features_all = torch.cat(point_features_list, dim=1)
#         batch_points = point_coords.view(batch_size, -1, 3)
#         batch_point_features = point_features_all.view(batch_size, -1, point_features_all.shape[-1])

#         with torch.no_grad():
#             pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
#                 batch_points, batch_point_features, rois
#             )  # pooled_features: (B, num_rois, num_sampled_points, 3 + C), pooled_empty_flag: (B, num_rois)

#             # canonical transformation
#             roi_center = rois[:, :, 0:3]
#             pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)

#             pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1])
#             pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
#                 pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
#             )
#             pooled_features[pooled_empty_flag.view(-1) > 0] = 0
#         return pooled_features

#     def forward(self, batch_dict):
#         """
#         Args:
#             batch_dict:

#         Returns:

#         """
#         targets_dict = self.proposal_layer(
#             batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
#         )
#         if self.training:
#             targets_dict = self.assign_targets(batch_dict)
#             batch_dict['rois'] = targets_dict['rois']
#             batch_dict['roi_labels'] = targets_dict['roi_labels']
#         print(batch_dict['rois'][0,0],batch_dict['rois'].shape,batch_dict['roi_labels'][0,0])
#         pooled_features = self.roipool3d_gpu(batch_dict)  # (total_rois, num_sampled_points, 3 + C)
#         print(pooled_features.shape)
#         xyz_input = pooled_features[..., 0:self.num_prefix_channels].transpose(1, 2).unsqueeze(dim=3).contiguous()
#         # xyz_features = self.xyz_up_layer(xyz_input)
#         point_features = pooled_features[..., self.num_prefix_channels:].transpose(1, 2).unsqueeze(dim=3)
#         # print(point_features.shape)
#         # Nrois * dimfeature * num_samples 
#         merged_features = point_features
#         # merged_features = torch.cat((xyz_features, point_features), dim=1)
#         # merged_features = self.merge_down_layer(merged_features)
#         #(B*Nrois,Nfeature,Nsamples,1)
#         # print(merged_features.shape)
#         src = merged_features.squeeze(3).permute(0,2,1)
#         # src = self.up_dimension(src)
#         pos = torch.zeros_like(src)
#         hs = self.transformer(src, self.query_embed.weight, pos)[0]
#         # l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [merged_features.squeeze(dim=3).contiguous()]

#         # for i in range(len(self.SA_modules)):
#         #     li_xyz, li_features, _ = self.SA_modules[i](l_xyz[i], l_features[i])
#         #     l_xyz.append(li_xyz)
#         #     l_features.append(li_features)
#         # print(hs.shape)
#         hs = hs.squeeze(0).permute(0,2,1)
#         shared_features = hs
#         # print(src.shape,self.query_embed.weight.shape,shared_features.shape)
#         # shared_features = l_features[-1]  # (total_rois, num_features, 1)
#         rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
#         rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
#         # print("277")
#         if not self.training:
#             batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
#                 batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
#             )
#             batch_dict['batch_cls_preds'] = batch_cls_preds
#             batch_dict['batch_box_preds'] = batch_box_preds
#             batch_dict['cls_preds_normalized'] = False
#         else:
#             targets_dict['rcnn_cls'] = rcnn_cls
#             targets_dict['rcnn_reg'] = rcnn_reg

#             self.forward_ret_dict = targets_dict
#         return batch_dict
import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.roipoint_pool3d import roipoint_pool3d_utils
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class PointRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        use_bn = self.model_cfg.USE_BN
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        self.num_prefix_channels = 3 + 2  # xyz + point_scores + point_depth
        xyz_mlps = [self.num_prefix_channels] + self.model_cfg.XYZ_UP_LAYER
        shared_mlps = []
        for k in range(len(xyz_mlps) - 1):
            shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        c_out = self.model_cfg.XYZ_UP_LAYER[-1]
        self.merge_down_layer = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, kernel_size=1, bias=not use_bn),
            *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
        )

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + self.model_cfg.SA_CONFIG.MLPS[k]

            npoint = self.model_cfg.SA_CONFIG.NPOINTS[k] if self.model_cfg.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModule(
                    npoint=npoint,
                    radius=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=use_bn
                )
            )
            channel_in = mlps[-1]

        self.cls_layers = self.make_fc_layers(
            input_channels=channel_in, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
            num_sampled_points=self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS,
            pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roipool3d_gpu(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:
        """
        batch_size = batch_dict['batch_size']
        batch_idx = batch_dict['point_coords'][:, 0]
        point_coords = batch_dict['point_coords'][:, 1:4]
        point_features = batch_dict['point_features']
        rois = batch_dict['rois']  # (B, num_rois, 7 + C)
        batch_cnt = point_coords.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert batch_cnt.min() == batch_cnt.max()

        point_scores = batch_dict['point_cls_scores'].detach()
        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER - 0.5
        point_features_list = [point_scores[:, None], point_depths[:, None], point_features]
        point_features_all = torch.cat(point_features_list, dim=1)
        batch_points = point_coords.view(batch_size, -1, 3)
        batch_point_features = point_features_all.view(batch_size, -1, point_features_all.shape[-1])

        with torch.no_grad():
            pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
                batch_points, batch_point_features, rois
            )  # pooled_features: (B, num_rois, num_sampled_points, 3 + C), pooled_empty_flag: (B, num_rois)

            # canonical transformation
            roi_center = rois[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)

            pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1])
            pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
            )
            pooled_features[pooled_empty_flag.view(-1) > 0] = 0
        return pooled_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
        Returns:
        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            targets_dict['calib'] = batch_dict['calib']
        pooled_features = self.roipool3d_gpu(batch_dict)  # (total_rois, num_sampled_points, 3 + C)

        xyz_input = pooled_features[..., 0:self.num_prefix_channels].transpose(1, 2).unsqueeze(dim=3).contiguous()
        xyz_features = self.xyz_up_layer(xyz_input)
        point_features = pooled_features[..., self.num_prefix_channels:].transpose(1, 2).unsqueeze(dim=3)
        merged_features = torch.cat((xyz_features, point_features), dim=1)
        merged_features = self.merge_down_layer(merged_features)

        l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [merged_features.squeeze(dim=3).contiguous()]
        # print(l_xyz[0])
        for i in range(len(self.SA_modules)):
            # print(l_xyz[i].shape,l_features[i].shape)
            li_xyz, li_features,_ = self.SA_modules[i](l_xyz[i], l_features[i])
            # if li_features is not None and li_xyz is not None:
                # print("*",li_xyz.shape,li_features.shape)
            # else:
                # print("*",li_xyz,li_features.shape)
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        shared_features = l_features[-1]  # (total_rois, num_features, 1)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict
        return batch_dict