'''
2D equivalent implementation of PointNet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)

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



class PlainNet_SEG(nn.Module):
    def __init__(self, input_channels = 3):
        super().__init__()
        self.channels = input_channels
        self.conv1 = nn.Conv2d(self.channels, 64, 1)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)
        self.conv4 = nn.Conv2d(64, 128, 1)
        self.conv5 = nn.Conv2d(128, 1024, 1)
        self.conv6 = nn.Conv2d(1091, 512, 1)
        self.conv7 = nn.Conv2d(512, 256, 1)
        self.conv8 = nn.Conv2d(256, 128, 1)
        self.conv9 = nn.Conv2d(128, 128, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)


    def forward(self, x, one_hot_vec):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        local_features = x # bchw

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        _, _, h, w = x.shape
        global_features = F.max_pool2d(x, (h, w))
        global_features = global_features.repeat(1, 1, h, w)
        one_hot_vec = one_hot_vec.view(-1, 3, 1, 1).repeat(1, 1, h, w)
        x = torch.cat([local_features, global_features, one_hot_vec], 1)
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        return x


if __name__ == '__main__':
    patch = torch.Tensor(2, 3, 32, 32)
    one_hot = torch.Tensor(2, 3)
    net = PlainNet(input_channels=3, layer_cfg=[128, -128, 256, 512], kernal_size=3, padding=1)
    print (net)
    print(sum([p.data.nelement() for p in net.parameters()]))
    output = net(patch)
    print(output.shape)

