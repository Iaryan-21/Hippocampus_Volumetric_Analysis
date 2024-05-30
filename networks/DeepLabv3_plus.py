import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import models

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        size = x.shape[2:]
        x1 = f.relu(self.bn1(self.conv1(x)))
        x2 = f.relu(self.bn2(self.conv2(x)))
        x3 = f.relu(self.bn3(self.conv3(x)))
        x4 = f.relu(self.bn4(self.conv4(x)))
        x5 = f.relu(self.global_avg_pool(x))
        x5 = f.interpolate(x5, size=size, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = f.relu(self.bn_out(self.conv_out(x)))
        return x

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = models.resnet101(pretrained=True)
        
        
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        self.decoder_conv1 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)  # Update input channels to 304
        self.decoder_bn1 = nn.BatchNorm2d(256)
        self.low_level_conv = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.low_level_bn = nn.BatchNorm2d(48)
        self.decoder_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.decoder_bn2 = nn.BatchNorm2d(256)
        self.decoder_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.decoder_bn3 = nn.BatchNorm2d(256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        size = x.shape[2:]
        low_level_features = self.backbone.layer1(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        x = self.backbone.layer4(self.backbone.layer3(self.backbone.layer2(low_level_features)))
        x = self.aspp(x)
        x = f.interpolate(x, size=(low_level_features.shape[2], low_level_features.shape[3]), mode='bilinear', align_corners=True)
        low_level_features = self.low_level_bn(self.low_level_conv(low_level_features))
        x = torch.cat([x, low_level_features], dim=1)  # Concatenate along the channel dimension
        x = f.relu(self.decoder_bn1(self.decoder_conv1(x)))
        x = f.relu(self.decoder_bn2(self.decoder_conv2(x)))
        x = f.relu(self.decoder_bn3(self.decoder_conv3(x)))
        x = f.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x = self.classifier(x)
        return x


# model = DeepLabV3Plus(num_classes=3, in_channels=1)
# print(model)
