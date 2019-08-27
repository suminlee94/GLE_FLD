import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

class VGG16Extractor(nn.Module):
    def __init__(self):
        super(VGG16Extractor, self).__init__()
        self.select = {
            '1': 'conv1_1',  # [batch_size, 64, 224, 224]
            '3': 'conv1_2',  # [batch_size, 64, 224, 224]
            '4': 'pooled_1',  # [batch_size, 64, 112, 112]
            '6': 'conv2_1',  # [batch_size, 128, 112, 112]
            '8': 'conv2_2',  # [batch_size, 128, 112, 112]
            '9': 'pooled_2',  # [batch_size, 128, 56, 56]
            '11': 'conv3_1',  # [batch_size, 256, 56, 56]
            '13': 'conv3_2',  # [batch_size, 256, 56, 56]
            '15': 'conv3_3',  # [batch_size, 256, 56, 56]
            '16': 'pooled_3',  # [batch_size, 256, 28, 28]
            '18': 'conv4_1',  # [batch_size, 512, 28, 28]
            '20': 'conv4_2',  # [batch_size, 512, 28, 28]
            '22': 'conv4_3',  # [batch_size, 512, 28, 28]
            '23': 'pooled_4',  # [batch_size, 512, 14, 14]
            '25': 'conv5_1',  # [batch_size, 512, 14, 14]
            '27': 'conv5_2',  # [batch_size, 512, 14, 14]
            '29': 'conv5_3',  # [batch_size, 512, 14, 14]
            '30': 'pooled_5',  # [batch_size , 512, 7, 7]
        }
        self.vgg = torchvision.models.vgg16(pretrained=True).features

    def forward(self, x):
        output = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                output[self.select[name]] = x
        return output

class NonLocal(nn.Module):
    def __init__(self, inplanes):
        super(NonLocal, self).__init__()
        self.inter_planes = int(inplanes / 2)
        self.g = nn.Conv2d(inplanes, self.inter_planes, 1, 1, 0)
        self.theta = nn.Conv2d(inplanes, self.inter_planes, 1, 1, 0)
        self.phi = nn.Conv2d(inplanes, self.inter_planes, 1, 1, 0)

        self.W = nn.Sequential(
            nn.Conv2d(self.inter_planes, inplanes, 1, 1, 0),
            nn.BatchNorm2d(inplanes)
        )
        nn.init.constant(self.W[1].weight, 1)
        nn.init.constant(self.W[1].bias, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_planes, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_planes, -1)
        theta_x = theta_x.permute(0, 2, 1)                              # (b, 784, 256)
        phi_x = self.phi(x).view(batch_size, self.inter_planes, -1)     # (b, 256, 196)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)                                  # (b, 784, 256)
        y = y.permute(0, 2, 1).contiguous()                             # (b, 256, 784)
        y = y.view(batch_size, self.inter_planes, *x.size()[2:])        # (b, 256, 28, 28)
        W_y = self.W(y)                                                 # (b, 512. 28, 28)
        z = W_y + x                                                     # (b, 512. 28, 28)

        return z

class GlobalLocalEmbedding(nn.Module):
    def __init__(self, in_channel):
        super(GlobalLocalEmbedding, self).__init__()
        self.non_local = NonLocal(in_channel)
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channel)

        self.conv2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.non_local(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(x)))
        return y

class LandmarkUpsample(nn.Module):
    def __init__(self, in_channel=256):
        super(LandmarkUpsample, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, 1, 1, 0)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv7 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv8 = nn.Conv2d(32, 32, 3, 1, 1)
        self.upconv3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.conv9 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv10 = nn.Conv2d(16, 8, 1, 1, 0)

        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.upconv1(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.upconv2(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.upconv3(x))
        x = self.relu(self.conv9(x))
        x = self.conv10(x)
        return x

class Network(nn.Module):
    def __init__(self, dataset, flag):
        super(Network, self).__init__()
        if dataset == 'fld':
            self.dataset = True
        else :
            self.dataset = False
        self.flag = flag

        self.feature_extractor = VGG16Extractor()
        self.upsampling = LandmarkUpsample(512)

        if self.flag:
            self.glem1 = GlobalLocalEmbedding(512)
            self.glem2 = GlobalLocalEmbedding(512)

            if self.dataset:
                self.glem3 = GlobalLocalEmbedding(512)

    def forward(self, sample):
        vgg16_output = self.feature_extractor(sample['image'])
        lm_feature = vgg16_output['conv4_3']

        if self.flag:
            lm_feature = self.glem1(lm_feature)
            lm_feature = self.glem2(lm_feature)

            if self.dataset:
                lm_feature = self.glem3(lm_feature)

        lm_pos_map = self.upsampling(lm_feature)

        return {'lm_pos_map' : lm_pos_map}




