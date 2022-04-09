import torch
import torch.nn as nn
import math
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack
from torch.nn.functional import grid_sample
import torch.nn.functional as F
BatchNorm2d = nn.BatchNorm2d
def knn(x, k):
    
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def transformer_neighbors(x, feature, k=20, idx=None):
    # print(x.shape,feature.shape)
    x = x.permute(0,2,1)
    '''
        input: x, [B,3,N]
               feature, [B,C,N]
        output: neighbor_x, [B,6,N,K]
                neighbor_feat, [B,2C,N,k]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_x = x.view(batch_size*num_points, -1)[idx, :]
    neighbor_x = neighbor_x.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    position_vector = (x - neighbor_x).permute(0, 3, 1, 2).contiguous() # B,3,N,k

    _, num_dims, _ = feature.size()

    feature = feature.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_feat = feature.view(batch_size*num_points, -1)[idx, :]
    neighbor_feat = neighbor_feat.view(batch_size, num_points, k, num_dims) 
    neighbor_feat = neighbor_feat.permute(0, 3, 1, 2).contiguous() # B,C,N,k
  
    return position_vector, neighbor_feat
class Point_Transformer(nn.Module):
    def __init__(self, input_features_dim):
        super(Point_Transformer, self).__init__()

        self.conv_theta1 = nn.Conv2d(3, input_features_dim, 1)
        self.conv_theta2 = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.bn_conv_theta = nn.BatchNorm2d(input_features_dim)

        self.conv_phi = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.conv_psi = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.conv_alpha = nn.Conv2d(input_features_dim, input_features_dim, 1)

        self.conv_gamma1 = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.conv_gamma2 = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.bn_conv_gamma = nn.BatchNorm2d(input_features_dim)

    def forward(self, xyz, features, k):

        position_vector, x_j = transformer_neighbors(xyz, features, k=k)

        delta = F.relu(self.bn_conv_theta(self.conv_theta2(self.conv_theta1(position_vector)))) # B,C,N,k
        # corrections for x_i
        x_i = torch.unsqueeze(features, dim=-1).repeat(1, 1, 1, k) # B,C,N,k

        linear_x_i = self.conv_phi(x_i) # B,C,N,k

        linear_x_j = self.conv_psi(x_j) # B,C,N,k

        relation_x = linear_x_i - linear_x_j + delta # B,C,N,k
        relation_x = F.relu(self.bn_conv_gamma(self.conv_gamma2(self.conv_gamma1(relation_x)))) # B,C,N,k

        weights = F.softmax(relation_x, dim=-1) # B,C,N,k
        features = self.conv_alpha(x_j) + delta # B,C,N,k

        f_out = weights * features # B,C,N,k
        f_out = torch.sum(f_out, dim=-1) # B,C,N

        return f_out


def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, stride=stride*2, kernel_size=3,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels*self.expansion,stride=stride, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels*self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride*2, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = F.relu(self.bn2(self.conv2(out)))
        # print(out.shape)
        out = self.bn3(self.conv3(out))
        # print(out.shape,self.shortcut(x).shape)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes )
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(outplanes, outplanes, 2*stride)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        return out
class BasicBlock2(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1):
        super(BasicBlock2, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes )
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(outplanes, outplanes, stride)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        return out
class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        #print(point_features.shape, img_features.shape)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features
#dot
class IA_layer3(nn.Module):
    def __init__(self, channels,dropout=0.5):
        print('##############ADDITION ATTENTION(ADD)#########')
        super(IA_layer3, self).__init__()
        self.ic,self.pc = channels
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.ic,self.pc)
        # self.fc1 = nn.Linear(self.pc,self.pc)
    def forward(self, img_feas, point_feas):

        batch = img_feas.size(0)
        # print(point_feas.shape,img_feas.shape)
        point_feas = point_feas.transpose(1,2).contiguous().view(-1, self.pc)
        # rp = self.fc1(point_feas)
        rp = point_feas
        img_feas = img_feas.transpose(1,2).contiguous().view(-1, self.ic)
        ri = self.fc2(img_feas)
        # ri = img_feas

        d = self.pc
        # img_feas = img_feas.permute(0,2,1)
        # print(point_feas.shape)
        rp = rp.view(batch,-1,d)
        ri = ri.view(batch,-1,d)

        rp1 = rp.clone()
        ri1 = ri.clone()
        # print(ri.shape,rp.shape)
        scores = torch.bmm(rp, ri.transpose(1, 2))/math.sqrt(d)
        # print(scores.shape)
        self.attention_weights = F.softmax(scores, dim=1)
        
        scores = torch.bmm(ri1, rp1.transpose(1, 2))/math.sqrt(d) 
        self.attention_weights1 = F.softmax(scores, dim=1)
        # print(self.attention_weights.shape,self.attention_weights1.shape,rp.shape,ri1.shape)
# class IA_layer3(nn.Module):
#     def __init__(self, channels,dropout=0.5):
#         print('##############ADDITION ATTENTION(ADD)#########')
#         super(IA_layer3, self).__init__()
#         self.ic,self.pc = channels
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(self.pc,self.ic)
    
#     def forward(self, img_feas, point_feas):

#         batch = img_feas.size(0)
#         point_feas = point_feas.transpose(1,2).contiguous().view(-1, self.pc)
#         rp = self.fc2(point_feas)
#         d = self.ic
#         img_feas = img_feas.permute(0,2,1)
#         # print(point_feas.shape)
#         rp = rp.view(batch,-1,d)
#         # print(ri.shape,point_feas.shape,d)

#         scores = torch.bmm(rp, img_feas.transpose(1, 2)) / math.sqrt(d)
#         self.attention_weights = F.softmax(scores, dim=1)

        # batch = img_feas.size(0)
        # img_fea_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic)
        # print(img_fea_f.shape,self.fc2)
        # ri = self.fc2(img_fea_f)
        # d = self.pc
        # point_feas = point_feas.permute(0,2,1)
        # # print(point_feas.shape)
        # ri = ri.view(batch,-1,d)
        # # print(ri.shape,point_feas.shape,d)

        # scores = torch.bmm(ri, point_feas.transpose(1, 2)) / math.sqrt(d)
        # self.attention_weights = F.softmax(scores, dim=1)

        # point_features = torch.bmm(self.dropout(self.attention_weights), rp)
        
        # img_feas = torch.bmm(self.dropout(self.attention_weights), ri)

        ####
        # scores2 = torch.bmm(point_feas, ri.transpose(1, 2)) / math.sqrt(d)
        # self.attention_weights2 = F.softmax(scores2, dim=1)
        # print(self.attention_weights.shape)
        # print(rp.shape,self.attention_weights.shape)
        return torch.bmm(self.dropout(self.attention_weights), rp),torch.bmm(self.dropout(self.attention_weights1), ri1)
class IA_layer2(nn.Module):
    def __init__(self, channels,dropout=0.5):
        print('##############ADDITION ATTENTION(ADD)#########')
        super(IA_layer2, self).__init__()
        self.ic,self.pc = channels
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.ic,self.pc)

    def forward(self, img_feas, point_feas):

        batch = img_feas.size(0)
        img_fea_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic)
        ri = self.fc2(img_fea_f)
        d = self.pc
        point_feas = point_feas.permute(0,2,1)
        # print(point_feas.shape)
        ri = ri.view(batch,-1,d)
        # print(ri.shape,point_feas.shape,d)

        scores = torch.bmm(ri, point_feas.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = F.softmax(scores, dim=1)

        ####
        # scores2 = torch.bmm(point_feas, ri.transpose(1, 2)) / math.sqrt(d)
        # self.attention_weights2 = F.softmax(scores2, dim=1)
        # print(self.attention_weights.shape)

        return torch.bmm(self.dropout(self.attention_weights), ri)
class Atten_Fusion_Conv2(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv2, self).__init__()

        self.IA_Layer = IA_layer2(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(2)
        # print(point_features.shape, img_features.shape)

        img_features =  self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)

        #fusion_features = img_features + point_features
        img_features = img_features.permute(0,2,1)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        return fusion_features
class Atten_Fusion_Conv3(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv3, self).__init__()

        self.IA_Layer = IA_layer3(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(3)
        # print(point_features.shape, img_features.shape)

        point_features,img_features =  self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)
        # print(point_features.shape,img_features.shape)
        #fusion_features = img_features + point_features
        point_features = point_features.permute(0,2,1)
        img_features = img_features.permute(0,2,1)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        return fusion_features
#cat
class IA_Layer(nn.Module):
    def __init__(self, channels):
        print('##############ADDITION ATTENTION(ADD)#########')
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                    nn.BatchNorm1d(self.pc),
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, self.ic)
        self.fc2 = nn.Linear(self.pc, self.pc)
        self.fc3 = nn.Linear(self.ic+self.pc, 2)


    def forward(self, img_feas, point_feas):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        # print(ri.shape,rp.shape)
        att = F.softmax(self.fc3(F.tanh(torch.cat([ri,rp],dim=1))))
        # att = F.sigmoid(self.fc3(F.tanh(ri + rp))) #BNx1
        # print(att.shape,ri.shape,rp.shape)
        # att = att.squeeze(1)

        att = att.view(batch, -1, 2) #B1N
        point_feas_f = point_feas_f.view(batch,att.shape[1],-1)
        img_feas_f = img_feas_f.view(batch,att.shape[1],-1)
        # print(point_feas_f.shape,img_feas_f.shape,att.shape)
        # print(torch.sum(att[:,:,0]).detach().cpu().numpy(),torch.sum(att[:,:,1]).detach().cpu().numpy())
        point_feas_f = point_feas_f * att[:,:,0].unsqueeze(2)
        img_feas_f =  img_feas_f * att[:,:,1].unsqueeze(2)
        # print(att.shape)
        # print(img_feas.size(), att.size())

        # img_feas_new = self.conv1(img_feas)
        # out = img_feas_new * att
        # print(img_feas_new)

        return point_feas_f,img_feas_f

class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_I, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        point_features,img_features =  self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)

        #fusion_features = img_features + point_features
        img_features = img_features.permute(0,2,1)
        point_features = point_features.permute(0,2,1)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features

#add
class IA_Layer1(nn.Module):
    def __init__(self, channels):
        print('##############ADDITION ATTENTION(ADD)#########')
        super(IA_Layer1, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                    nn.BatchNorm1d(self.pc),
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)


    def forward(self, img_feas, point_feas):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        # print(ri.shape,rp.shape)
        att = F.sigmoid(self.fc3(F.tanh(ri + rp))) #BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1) #B1N
        # print(img_feas.size(), att.size())

        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att

        return out
class Atten_Fusion_Conv1(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv1, self).__init__()

        self.IA_Layer = IA_Layer1(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        img_features =  self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)

        #fusion_features = img_features + point_features
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features

def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """

    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)
    # print(feature_map.shape,xy.shape)
    interpolate_feature = grid_sample(feature_map, xy)  # (B,C,1,N)

    return interpolate_feature.squeeze(2) # (B,C,N)
class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        cfg = model_cfg
        # print(cfg)
        self.SA_modules = nn.ModuleList()
        self.pointformer = nn.ModuleList()
        channel_in = input_channels - 5
        # channel_in = 0
        self.num_points_each_layer = []
        featurechannel = [160,384,768,1536]
        skip_channel_list = [input_channels - 5]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            # self.pointformer.append(
            #     Point_Transformer(input_features_dim=featurechannel[k])
            # )
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        
        
        if cfg.LI_FUSION.ENABLED:
            self.Img_Block = nn.ModuleList()
            self.Fusion_Conv = nn.ModuleList()
            self.DeConv = nn.ModuleList()
            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                
                if cfg.LI_FUSION.ADD_Image_Attention:
                    # if i!=len(cfg.LI_FUSION.IMG_CHANNELS) - 2:
                    #     self.Fusion_Conv.append(
                    #        None)
                    #     self.Img_Block.append(BasicBlock(cfg.LI_FUSION.IMG_CHANNELS[i], cfg.LI_FUSION.IMG_CHANNELS[i+1], stride=1))
                    # else:

                    self.Fusion_Conv.append(
                        Atten_Fusion_Conv1(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.POINT_CHANNELS[i],
                                        cfg.LI_FUSION.POINT_CHANNELS[i]))
                    self.Img_Block.append(BasicBlock(cfg.LI_FUSION.IMG_CHANNELS[i], cfg.LI_FUSION.IMG_CHANNELS[i+1], stride=1))
                    # print(self.Img_Block[i])
                else:
                    self.Fusion_Conv.append(Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + cfg.LI_FUSION.POINT_CHANNELS[i],
                                                        cfg.LI_FUSION.POINT_CHANNELS[i]))

                # self.DeConv.append(nn.ConvTranspose2d(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.DeConv_Reduce[i],
                #                                   kernel_size=cfg.LI_FUSION.DeConv_Kernels[i],
                #                                   stride=cfg.LI_FUSION.DeConv_Kernels[i]))
                # print(self.DeConv[i])
            self.image_fusion_conv = nn.Conv2d(sum(cfg.LI_FUSION.DeConv_Reduce), cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, kernel_size = 1)
            self.image_fusion_bn = torch.nn.BatchNorm2d(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4)

            if cfg.LI_FUSION.ADD_Image_Attention:
                self.final_fusion_img_point = Atten_Fusion_Conv1(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, cfg.LI_FUSION.IMG_FEATURES_CHANNEL, cfg.LI_FUSION.IMG_FEATURES_CHANNEL)
            else:
                self.final_fusion_img_point = Fusion_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL + cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, cfg.LI_FUSION.IMG_FEATURES_CHANNEL)


        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )
        
        # self.IMG_CHANNELS = [3, 64, 128, 256, 512]
        # self.img_block = nn.ModuleList()
        # for i in range(len(self.IMG_CHANNELS)-1):
        #     self.img_block.append(BasicBlock(self.IMG_CHANNELS[i],self.IMG_CHANNELS[i+1], stride=1))
        
        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        # features = None
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        # print(batch_dict['co_xy'].shape,batch_dict['points'].shape)
        cfg = self.model_cfg
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points'][:,:4]
        
        batch_idx, xyz, features = self.break_up_pc(points)
        
        # print(xyz.shape,features,"111")
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        # print(batch_dict['points'])
        # xy = batch_dict['co_xy']
        xy = batch_dict['points'][:,4:]
        # print(points.shape,xy.shape)
        image = batch_dict['images']
        import cv2
        import numpy as np
        # print("?????",image[0].detach().cpu().numpy().astype(np.int))
        # print(image[0].shape)
        # cv2.imwrite("img2000000000000000.png",image[0].detach().cpu().permute(1,2,0).numpy())
        # image = image.permute(0,1,3,2)
        xy = xy.view(batch_size, -1, 2)
        # print(xy)
        # print(image.shape)
        # print(batch_dict['points'])
        if cfg.LI_FUSION.ENABLED:
            #### normalize xy to [-1,1]
            size_range = [1280.0, 384.0]
            xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
            xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0  # = xy / (size_range - 1.) * 2 - 1.
            l_xy_cor = [xy]
            img = [image]
        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)

        xyt = xy.unsqueeze(1)
        # print(image.shape,xyt.shape)
        # pointsrgb = grid_sample(image, xyt, mode='nearest')
        # # # print(pointsrgb.shape,xyz.shape,"244")
        # pointsrgb = torch.cat((xyz,pointsrgb.squeeze(2).permute(0,2,1)),dim=2)
        # pointsrgb = pointsrgb.detach().cpu().numpy()
        # # print(pointsrgb)
        # np.savetxt("pointsrgb.txt",pointsrgb[0])
        # print(image.shape,xy.shape,xyz.shape)
        # print(xyz.shape,"****",batch_dict['co_xy'].shape,batch_dict['images'].shape)
        # features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None
        # print(xyz.shape,features)
        l_xyz, l_features = [xyz], [None]
        # print(img[0].shape)
        # print(l_xyz[0].shape,l_features,img,l_xy_cor)
        for i in range(len(self.SA_modules)):
            # print(l_features[i].shape)
            li_xyz, li_features,li_index = self.SA_modules[i](l_xyz[i], l_features[i])
            # print(li_xyz.shape,li_features.shape,li_index.shape)
            # print(li_features.shape,"269")
            if cfg.LI_FUSION.ENABLED:

                li_index = li_index.long().unsqueeze(-1).repeat(1,1,2)
                # print(l_xy_cor[i],li_index)
                li_xy_cor = torch.gather(l_xy_cor[i],1,li_index)
                
                image = self.Img_Block[i](img[i])
                # print(image.shape)
                # print(image.shape,img[i].shape)
                # print(image.shape,li_xy_cor.shape)
                img_gather_feature = Feature_Gather(image,li_xy_cor) #, scale= 2**(i+1))
                # print(img_gather_feature.shape,"279")
                # featurecat = torch.cat((li_features,img_gather_feature),dim=1)
                # featurecat = self.pointformer[i](li_xyz,featurecat,20)

                # li_features = featurecat[:,:li_features.shape[1],:]
                # img_gather_feature = featurecat[:,li_features.shape[1]:,:]
                # print(li_features.shape,img_gather_feature.shape)
                if i == len(self.SA_modules)-1:
                    li_features = self.Fusion_Conv[i](li_features,img_gather_feature)
                # print(li_features.shape,"280")
                l_xy_cor.append(li_xy_cor)
                img.append(image)

            # l_xyz.append(li_xyz)
            # l_features.append(li_features)
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)
            # print(l_features[i-1].shape)

        # if cfg.LI_FUSION.ENABLED:
        #     #for i in range(1,len(img))
        #     DeConv = []
        #     for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
        #         DeConv.append(self.DeConv[i](img[i + 1]))
        #     # print(DeConv)
        #     # for de in DeConv:
        #     #     print(de.shape)
        #     de_concat = torch.cat(DeConv,dim=1)
        
        #     img_fusion = F.relu(self.image_fusion_bn(self.image_fusion_conv(de_concat)))
        #     img_fusion_gather_feature = Feature_Gather(img_fusion, xy)
        #     l_features[0] = self.final_fusion_img_point(l_features[0], img_fusion_gather_feature)

        point_features = l_features[0].permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
        # print(batch_dict.keys(),"output",batch_dict['point_features'].shape,batch_dict['point_coords'].shape)
        return batch_dict


class PointNet2Backbone(nn.Module):
    """
    DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723
    """
    def __init__(self, model_cfg, input_channels, **kwargs):
        assert False, 'DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723'
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3



        self.num_points_each_layer = []
        skip_channel_list = [input_channels]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            self.num_points_each_layer.append(self.model_cfg.SA_CONFIG.NPOINTS[k])
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules_stack.StackSAModuleMSG(
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out
    
        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules_stack.StackPointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        # features = None
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        # print(xyz.shape,features,"213")
        # print(xyz.shape,features.shape)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        l_xyz, l_features, l_batch_cnt = [xyz], [features], [xyz_batch_cnt]
        for i in range(len(self.SA_modules)):
            new_xyz_list = []
            for k in range(batch_size):
                if len(l_xyz) == 1:
                    cur_xyz = l_xyz[0][batch_idx == k]
                else:
                    last_num_points = self.num_points_each_layer[i - 1]
                    cur_xyz = l_xyz[-1][k * last_num_points: (k + 1) * last_num_points]
                cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(
                    cur_xyz[None, :, :].contiguous(), self.num_points_each_layer[i]
                ).long()[0]
                if cur_xyz.shape[0] < self.num_points_each_layer[i]:
                    empty_num = self.num_points_each_layer[i] - cur_xyz.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                new_xyz_list.append(cur_xyz[cur_pt_idxs])
            new_xyz = torch.cat(new_xyz_list, dim=0)

            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(self.num_points_each_layer[i])
            li_xyz, li_features = self.SA_modules[i](
                xyz=l_xyz[i], features=l_features[i], xyz_batch_cnt=l_batch_cnt[i],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )



            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_batch_cnt.append(new_xyz_batch_cnt)

        l_features[0] = points[:, 1:]
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                unknown=l_xyz[i - 1], unknown_batch_cnt=l_batch_cnt[i - 1],
                known=l_xyz[i], known_batch_cnt=l_batch_cnt[i],
                unknown_feats=l_features[i - 1], known_feats=l_features[i]
            )

        batch_dict['point_features'] = l_features[0]
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0]), dim=1)
        return batch_dict
# import torch
# import torch.nn as nn

# from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
# from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
# from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack


# class PointNet2MSG(nn.Module):
#     def __init__(self, model_cfg, input_channels, **kwargs):
#         super().__init__()
#         self.model_cfg = model_cfg

#         self.SA_modules = nn.ModuleList()
#         channel_in = input_channels - 3

#         self.num_points_each_layer = []
#         skip_channel_list = [input_channels - 3]
#         for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
#             mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
#             channel_out = 0
#             for idx in range(mlps.__len__()):
#                 mlps[idx] = [channel_in] + mlps[idx]
#                 channel_out += mlps[idx][-1]

#             self.SA_modules.append(
#                 pointnet2_modules.PointnetSAModuleMSG(
#                     npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
#                     radii=self.model_cfg.SA_CONFIG.RADIUS[k],
#                     nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
#                     mlps=mlps,
#                     use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
#                 )
#             )
#             skip_channel_list.append(channel_out)
#             channel_in = channel_out

#         self.FP_modules = nn.ModuleList()

#         for k in range(self.model_cfg.FP_MLPS.__len__()):
#             pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
#             self.FP_modules.append(
#                 pointnet2_modules.PointnetFPModule(
#                     mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
#                 )
#             )

#         self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

#     def break_up_pc(self, pc):
#         batch_idx = pc[:, 0]
#         xyz = pc[:, 1:4].contiguous()
#         features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
#         return batch_idx, xyz, features

#     def forward(self, batch_dict):
#         """
#         Args:
#             batch_dict:
#                 batch_size: int
#                 vfe_features: (num_voxels, C)
#                 points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
#         Returns:
#             batch_dict:
#                 encoded_spconv_tensor: sparse tensor
#                 point_features: (N, C)
#         """
#         batch_size = batch_dict['batch_size']
#         points = batch_dict['points']
#         batch_idx, xyz, features = self.break_up_pc(points)

#         xyz_batch_cnt = xyz.new_zeros(batch_size).int()
#         for bs_idx in range(batch_size):
#             xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

#         assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
#         xyz = xyz.view(batch_size, -1, 3)
#         features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None

#         l_xyz, l_features = [xyz], [features]
#         for i in range(len(self.SA_modules)):
#             li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
#             l_xyz.append(li_xyz)
#             l_features.append(li_features)

#         for i in range(-1, -(len(self.FP_modules) + 1), -1):
#             l_features[i - 1] = self.FP_modules[i](
#                 l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
#             )  # (B, C, N)

#         point_features = l_features[0].permute(0, 2, 1).contiguous()  # (B, N, C)
#         batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
#         batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
#         return batch_dict


# class PointNet2Backbone(nn.Module):
#     """
#     DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723
#     """
#     def __init__(self, model_cfg, input_channels, **kwargs):
#         assert False, 'DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723'
#         super().__init__()
#         self.model_cfg = model_cfg

#         self.SA_modules = nn.ModuleList()
#         channel_in = input_channels - 3

#         self.num_points_each_layer = []
#         skip_channel_list = [input_channels]
#         for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
#             self.num_points_each_layer.append(self.model_cfg.SA_CONFIG.NPOINTS[k])
#             mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
#             channel_out = 0
#             for idx in range(mlps.__len__()):
#                 mlps[idx] = [channel_in] + mlps[idx]
#                 channel_out += mlps[idx][-1]

#             self.SA_modules.append(
#                 pointnet2_modules_stack.StackSAModuleMSG(
#                     radii=self.model_cfg.SA_CONFIG.RADIUS[k],
#                     nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
#                     mlps=mlps,
#                     use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
#                 )
#             )
#             skip_channel_list.append(channel_out)
#             channel_in = channel_out

#         self.FP_modules = nn.ModuleList()

#         for k in range(self.model_cfg.FP_MLPS.__len__()):
#             pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
#             self.FP_modules.append(
#                 pointnet2_modules_stack.StackPointnetFPModule(
#                     mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
#                 )
#             )

#         self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

#     def break_up_pc(self, pc):
#         batch_idx = pc[:, 0]
#         xyz = pc[:, 1:4].contiguous()
#         features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
#         return batch_idx, xyz, features

#     def forward(self, batch_dict):
#         """
#         Args:
#             batch_dict:
#                 batch_size: int
#                 vfe_features: (num_voxels, C)
#                 points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
#         Returns:
#             batch_dict:
#                 encoded_spconv_tensor: sparse tensor
#                 point_features: (N, C)
#         """
#         batch_size = batch_dict['batch_size']
#         points = batch_dict['points']
#         batch_idx, xyz, features = self.break_up_pc(points)

#         xyz_batch_cnt = xyz.new_zeros(batch_size).int()
#         for bs_idx in range(batch_size):
#             xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

#         l_xyz, l_features, l_batch_cnt = [xyz], [features], [xyz_batch_cnt]
#         for i in range(len(self.SA_modules)):
#             new_xyz_list = []
#             for k in range(batch_size):
#                 if len(l_xyz) == 1:
#                     cur_xyz = l_xyz[0][batch_idx == k]
#                 else:
#                     last_num_points = self.num_points_each_layer[i - 1]
#                     cur_xyz = l_xyz[-1][k * last_num_points: (k + 1) * last_num_points]
#                 cur_pt_idxs = pointnet2_utils_stack.farthest_point_sample(
#                     cur_xyz[None, :, :].contiguous(), self.num_points_each_layer[i]
#                 ).long()[0]
#                 if cur_xyz.shape[0] < self.num_points_each_layer[i]:
#                     empty_num = self.num_points_each_layer[i] - cur_xyz.shape[1]
#                     cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
#                 new_xyz_list.append(cur_xyz[cur_pt_idxs])
#             new_xyz = torch.cat(new_xyz_list, dim=0)

#             new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(self.num_points_each_layer[i])
#             li_xyz, li_features = self.SA_modules[i](
#                 xyz=l_xyz[i], features=l_features[i], xyz_batch_cnt=l_batch_cnt[i],
#                 new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
#             )

#             l_xyz.append(li_xyz)
#             l_features.append(li_features)
#             l_batch_cnt.append(new_xyz_batch_cnt)

#         l_features[0] = points[:, 1:]
#         for i in range(-1, -(len(self.FP_modules) + 1), -1):
#             l_features[i - 1] = self.FP_modules[i](
#                 unknown=l_xyz[i - 1], unknown_batch_cnt=l_batch_cnt[i - 1],
#                 known=l_xyz[i], known_batch_cnt=l_batch_cnt[i],
#                 unknown_feats=l_features[i - 1], known_feats=l_features[i]
#             )

#         batch_dict['point_features'] = l_features[0]
#         batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0]), dim=1)
#         return batch_dict
