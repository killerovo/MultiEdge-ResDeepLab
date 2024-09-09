import math
import os
import pdb
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax, interpolate
import torch.nn.functional as F


#########################################
#                                       #
#   ResNet for DeepLab v3+ Backbone     #
#                                       #
#########################################
def extract_edges_gpu(batch_images):
    # Define Sobel filters for edge detection
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=batch_images.device).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32, device=batch_images.device).unsqueeze(0).unsqueeze(0)

    # Apply Sobel filters to detect edges
    # Add a channel dimension to batch_images
    batch_images = batch_images.unsqueeze(1)  # Shape: (batch, 1, H, W)

    # Convolve with Sobel filters using 'same' padding to keep output size unchanged
    edges_x = F.conv2d(batch_images, sobel_x, padding=1)  # Horizontal edges
    edges_y = F.conv2d(batch_images, sobel_y, padding=1)  # Vertical edges

    # Combine edges (magnitude of gradients)
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

    # Normalize the output to be in the range [0, 1]
    edges = edges.squeeze(1)  # Remove the channel dimension to keep shape (batch, H, W)

    # Clip to [0, 1] for any overflow values
    edges = torch.clamp(edges, 0, 1)

    return edges


# 原始模态图像的拼接
def concatenate_modalities(t1, t1ce, t2):
    combined_input = torch.cat((t1, t1ce, t2), dim=1)  # 在通道维度拼接原始模态
    return combined_input


# def save_images(image_tensor, edges_tensor, folder, prefix='image'):
#     os.makedirs(folder, exist_ok=True)  # Create the directory if it doesn't exist
#     batch_size = image_tensor.size(0)

#     for i in range(batch_size):
#         original_image = image_tensor[i].cpu().numpy()
#         edge_image = edges_tensor[i].cpu().numpy()

#         fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#         axs[0].imshow(original_image, cmap='gray')
#         axs[0].set_title('Original')
#         axs[0].axis('off')

#         axs[1].imshow(edge_image, cmap='gray')
#         axs[1].set_title('Edge')
#         axs[1].axis('off')

#         plt.savefig(os.path.join(folder, f"{prefix}_{i}.png"))
#         plt.close()

# def save_prediction_images(prediction_tensor, folder, prefix='prediction'):
#     os.makedirs(folder, exist_ok=True)  # Create the directory if it doesn't exist
#     batch_size = prediction_tensor.size(0)

#     for i in range(batch_size):
#         # Detach the tensor from the computation graph and move it to CPU, then convert to NumPy
#         prediction = prediction_tensor[i].detach().cpu().numpy()  # Ensure tensor is detached
#         prediction = np.argmax(prediction, axis=0)  # 获取类别的最大值作为预测结果

#         plt.figure(figsize=(6, 6))
#         plt.imshow(prediction, cmap='gray')  # 显示预测结果
#         plt.axis('off')
#         plt.savefig(os.path.join(folder, f"{prefix}_{i}.png"))
#         plt.close()


# 边缘图像与模态图像的拼接
def concatenate_edge_and_modalities(t1, t1ce, t2):
    # 提取每个模态的边缘信息
    edge_t1 = extract_edges_gpu(t1)
    edge_t1ce = extract_edges_gpu(t1ce)
    edge_t2 = extract_edges_gpu(t2)
    # # Save original and edge-detected images
    # save_images(t1, edge_t1, folder='output', prefix='T1')
    # save_images(t1ce, edge_t1ce, folder='output', prefix='T1CE')
    # save_images(t2, edge_t2, folder='output', prefix='T2')

    # # 在通道维度拼接边缘信息
    combined_edge_input = torch.cat((edge_t1.unsqueeze(1), edge_t1ce.unsqueeze(1), edge_t2.unsqueeze(1)), dim=1)
    # import pdb;pdb.set_trace()
    return combined_edge_input


def conv1x1(in_planes, out_planes, drop_rate=0):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.Dropout2d(p=drop_rate),
        nn.ReLU(inplace=True),
    )
    return model


def conv3x3(in_planes, out_planes, drop_rate=0):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.Dropout2d(p=drop_rate),
        nn.ReLU(inplace=True),
    )
    return model


def conv3x3_simple(in_planes, out_planes, stride=1, drop_rate=0):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, drop_rate=0):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=1, padding=2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.Dropout2d(p=drop_rate),
        nn.ReLU(inplace=True),
    )
    return model


def atrous_conv(in_planes, out_planes, atrous_rate, drop_rate=0):
    model = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                  padding=atrous_rate, dilation=atrous_rate, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.Dropout2d(p=drop_rate),
        nn.ReLU(inplace=True),
    )
    return model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0):
        super(BasicBlock, self).__init__()
        self.dp = nn.Dropout2d(p=drop_rate)
        self.conv1 = conv3x3_simple(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_simple(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dp(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.dp(out)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0):
        super(Bottleneck, self).__init__()
        self.dp = nn.Dropout2d(p=drop_rate)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dp(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dp(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.dp(out)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, drop_rate=0):
        super(ResNet, self).__init__()
        self.expansion = block.expansion
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_rate=drop_rate)

        self.low_conv = nn.Conv2d(64 * block.expansion, 64, kernel_size=3,
                                  stride=1, padding=1, bias=False)
        self.atr_conv1 = atrous_conv(256 * block.expansion, 128, 1)
        self.atr_conv2 = atrous_conv(256 * block.expansion, 128, 2)
        self.atr_conv3 = atrous_conv(256 * block.expansion, 128, 4)
        self.atr_conv4 = atrous_conv(256 * block.expansion, 128, 16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, drop_rate=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                nn.Dropout2d(p=drop_rate)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate=drop_rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_rate=drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        low_x = self.layer1(x)
        x = self.layer2(low_x)
        x = self.layer3(x)
        x1 = self.atr_conv1(x)
        x2 = self.atr_conv2(x)
        x3 = self.atr_conv3(x)
        x4 = self.atr_conv4(x)
        if self.expansion != 1:
            low_x = self.low_conv(low_x)
        return torch.cat((x1, x2, x3, x4), 1), low_x


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnet_test():
    net = resnet152()
    net = nn.DataParallel(net)
    net.cuda()

    x = torch.randn((1, 1, 1024, 1024))
    y1, y2 = net(x)
    print(y1.size())
    print(y2.size())


# resnet_test()


#########################################
#                                       #
#           DeepLab V3 +                #
#                                       #
#########################################
class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchsyncbn'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class Encoder(nn.Module):

    def __init__(self, drop_rate=0):
        super(Encoder, self).__init__()
        self.dp = nn.Dropout2d(p=drop_rate)
        self.conv1 = conv1x1(1024, 256)
        self.atr_conv1 = atrous_conv(1024, 256, 6, drop_rate)
        self.atr_conv2 = atrous_conv(1024, 256, 12, drop_rate)
        self.atr_conv3 = atrous_conv(1024, 256, 18, drop_rate)
        self.avgpool = nn.AvgPool2d(2)  # impossible due to 15/2 = 7.5
        self.conv2 = conv1x1(1024, 256)
        self.conv_cat = conv1x1(1024, 512)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.atr_conv1(x)
        y3 = self.atr_conv2(x)
        y4 = self.atr_conv3(x)
        y5 = self.avgpool(x)
        y5 = self.conv2(y5)
        y5 = interpolate(y5, scale_factor=2, mode='bilinear', align_corners=True)

        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.conv_cat(y)
        return interpolate(y, scale_factor=4, mode='bilinear', align_corners=True)


class Deeplab_V3_Plus(nn.Module):

    def __init__(self, class_num=4, drop_rate=0):
        super(Deeplab_V3_Plus, self).__init__()

        self.resnet = resnet50(drop_rate=drop_rate)
        self.resnet_edge = resnet50(drop_rate=drop_rate)
        self.conv0 = conv1x1(128, 256, drop_rate)
        self.encoder = Encoder(drop_rate)
        self.conv1 = conv3x3(768, 512, drop_rate)
        self.cls_head = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(512, class_num, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x, ground_truth=None):
        # ResNet backbone to extract features
        y, low_y = self.resnet(x)

        # Extract edges from modalities
        input_edges = concatenate_edge_and_modalities(x[:, 0], x[:, 1], x[:, 2])
        y_edge, low_y_edge = self.resnet_edge(input_edges)

        # Combine low-level features
        low_y = self.conv0(torch.cat([low_y, low_y_edge], dim=1))
        y = self.encoder(torch.cat([y, y_edge], dim=1))
        y = torch.cat([low_y, y], dim=1)
        y = self.conv1(y)

        # Prediction head
        y = self.cls_head(y)
        y = interpolate(y, scale_factor=4, mode='bilinear', align_corners=True)
        label = softmax(y, dim=1)

        return label

