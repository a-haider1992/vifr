from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch
import tqdm
from torch import nn

###########
from .irse import IResNet


class SPPModule(nn.Module):
    def __init__(self, pool_mode='avg', sizes=(1, 2, 3, 6)):
        super().__init__()
        if pool_mode == 'avg':
            pool_layer = nn.AdaptiveAvgPool2d
        elif pool_mode == 'max':
            pool_layer = nn.AdaptiveMaxPool2d
        else:
            raise NotImplementedError

        self.pool_blocks = nn.ModuleList([
            nn.Sequential(pool_layer(size), nn.Flatten()) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.pool_blocks]
        x = torch.cat(xs, dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x


# Gender Feature Extractor
class GenderFeatureExtractor(nn.Module):
    def __init__(self):
        super(GenderFeatureExtractor, self).__init__()

        # Define convolutional layers
        # self.conv1 = nn.Conv2d(
        #     in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(
        #     in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(
        #     in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(
        #     in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Define pooling layers
        pool_size = (7, 7)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.pool = nn.AdaptiveMaxPool2d(pool_size)
        self.bn = nn.BatchNorm2d(num_features=512)

        # Define fully connected layers
        # self.fc1 = nn.Linear(in_features=256 * 16 * 16, out_features=1024)
        self.fc2 = nn.Linear(in_features=512 * 7 * 7, out_features=2)


    def forward(self, x):
        x = self.pool(x)
        x= self.bn(x)

        # Flatten output tensor for fully connected layers
        x = x.view(-1, 512 * 7 * 7)

        # # Pass input through fully connected layer
        x = self.fc2(x)
        return x

# Ethinicty Classifier
class EthnicityFeatureExtractor(nn.Module):
    def __init__(self):
        super(EthnicityFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.pool1 = nn.AdaptiveMaxPool2d((64, 64))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.AdaptiveMaxPool2d((128, 128))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool3 = nn.AdaptiveMaxPool2d((512, 512))
        # self.fc1 = nn.Linear(64 * 6 * 6, 128)
        # self.fc2 = nn.Linear(128, 5)  # 5 possible ethnicities

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = self.pool3(nn.functional.relu(self.conv3(x)))
        # x = x.view(-1, 64 * 6 * 6)
        # x = nn.functional.relu(self.fc1(x))
        # x = self.fc2(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, channels=512, reduction=16):
        super(AttentionModule, self).__init__()
        kernel_size = 7
        pool_size = (1, 2, 3)
        self.avg_spp = SPPModule('avg', pool_size)
        self.max_spp = SPPModule('max', pool_size)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                      dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

        _channels = channels * int(sum([x ** 2 for x in pool_size]))
        self.channel = nn.Sequential(
            nn.Conv2d(
                _channels, _channels // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                _channels // reduction, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_input = self.avg_spp(x) + self.max_spp(x)
        channel_scale = self.channel(channel_input)

        spatial_input = torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        spatial_scale = self.spatial(spatial_input)

        x_age = (x * channel_scale + x * spatial_scale) * 0.5

        x_id = x - x_age

        return x_id, x_age


def twinify_tensors(tensor1, tensor2):
        if tensor1.shape[2:] != tensor2.shape[2:]:
            h_diff = abs(tensor1.shape[2] - tensor2.shape[2])
            w_diff = abs(tensor1.shape[3] - tensor2.shape[3])
            if tensor1.shape[2] > tensor2.shape[2]:
                tensor2 = torch.nn.functional.pad(tensor2, 
                                                (0, 0, 0, 0, h_diff // 2, h_diff // 2 + 
                                                h_diff % 2, w_diff // 2, w_diff // 2 + 
                                                w_diff % 2))
            else:
                tensor1 = torch.nn.functional.pad(tensor1, 
                                                (0, 0, 0, 0, h_diff // 2, h_diff // 2 + 
                                                h_diff % 2, w_diff // 2, w_diff // 2 + 
                                                w_diff % 2))
            return tensor1, tensor2

class AIResNet(IResNet):
    def __init__(self, input_size, num_layers, mode='ir', **kwargs):
        super(AIResNet, self).__init__(input_size, num_layers, mode)
        self.fsm = AttentionModule()
        self.channel_reducer = torch.nn.Conv2d(in_channels=512+256+128+64, out_channels=512, kernel_size=1)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(512 * (input_size // 16) ** 2, 512),
            nn.BatchNorm1d(512))
        self._initialize_weights()

    def forward(self, x, return_age=False, return_gender=False, return_shortcuts=False):
        x_1 = self.input_layer(x)
        x_2 = self.block1(x_1)
        x_3 = self.block2(x_2)
        x_4 = self.block3(x_3)
        x_5 = self.block4(x_4)
        # Id and age related features by Attention Module
        x_id, x_age = self.fsm(x_5)

        # Embedding of image
        embedding = self.output_layer(x_id)

        # Gender features from Gender Module
        # Upsample each tensor assuming tensors are of shape (batch_size, channels, width, height)
        # assert x_2.ndim == 4, f"Expected tensor to have four dimensions, but got {x_2.ndim}"
        # assert x_3.ndim == 4, f"Expected tensor to have four dimensions, but got {x_3.ndim}"
        # assert x_4.ndim == 4, f"Expected tensor to have four dimensions, but got {x_4.ndim}"
        # assert x_5.ndim == 4, f"Expected tensor to have four dimensions, but got {x_5.ndim}"

        # _, up_x_5 = twinify_tensors(x_2, x_5)
        # _, up_x_4 = twinify_tensors(x_2, x_4)
        # _, up_x_3 = twinify_tensors(x_2, x_3)

        # up_x_5 = F.interpolate(x_5, size=(56, 56), mode='bilinear', align_corners=False)
        # up_x_4 = F.interpolate(x_4, size=(56, 56), mode='bilinear', align_corners=False)
        # up_x_3 = F.interpolate(x_3, size=(56, 56), mode='bilinear', align_corners=False)

        ## Concate along channels
        # concatenated_x = torch.cat([x_2, up_x_3, up_x_4, up_x_5], dim=1)
        # Concate along height
        # concatenated_x = torch.cat([concatenated_x, x_2, up_x_3, up_x_4, up_x_5], dim=2)
        # Concate along width
        # concatenated_x = torch.cat([concatenated_x, x_2, up_x_3, up_x_4, up_x_5], dim=3)

        # Downsample the concatenated tensor for substraction
        # concatenated_x = F.interpolate(concatenated_x, size=(7, 7), mode='bicubic', align_corners=False)
        #  channel-wise pooling
        # concatenated_x = self.channel_reducer(concatenated_x)
        # concatenated_x = concatenated_x[:, :512, :, :]
        # concatenated_x = F.interpolate(concatenated_x, size=(512, 512), mode='trilinear', align_corners=True)

        # print(f'The final concatenated tensor shape:{concatenated_x.shape}')

        # Approach 1
        # x_gender = concatenated_x - (x_age + x_id)
        # Approach 2
        # x_gender = x - x_5
        
        if return_shortcuts:
            return x_1, x_2, x_3, x_4, x_5, x_id, x_age
        if return_age:
            return embedding, x_id, x_age
        # if return_gender:
        #     return embedding, x_id, x_age, x_gender
        return embedding


class AgeEstimationModule(nn.Module):
    def __init__(self, input_size, age_group, dist=False):
        super(AgeEstimationModule, self).__init__()
        out_neurons = 101
        self.age_output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512 * (input_size // 16) ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True) if dist else nn.ReLU(inplace=True),
            nn.Linear(512, out_neurons),
        )
        self.group_output_layer = nn.Linear(out_neurons, age_group)

    def forward(self, x_age):
        x_age = self.age_output_layer(x_age)
        x_group = self.group_output_layer(x_age)
        return x_age, x_group


backbone_dict = {
    'ir34': partial(AIResNet, num_layers=[3, 4, 6, 3], mode="ir"),
    'ir50': partial(AIResNet, num_layers=[3, 4, 14, 3], mode="ir"),
    'ir64': partial(AIResNet, num_layers=[3, 4, 10, 3], mode="ir"),
    'ir101': partial(AIResNet, num_layers=[3, 13, 30, 3], mode="ir"),
    'irse101': partial(AIResNet, num_layers=[3, 13, 30, 3], mode="ir_se")
}
