import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from  torchvision.models import alexnet
from  torchvision import models

class Residualblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out+residual)


class TinyResNet(nn.Module):
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.block1 = Residualblock(8, 8)
        self.block2 = Residualblock(8, 8)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)  # Tiny fc layer
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
class TinyResNet1C(nn.Module):
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.block1 = Residualblock(8, 8)
        self.block2 = Residualblock(8, 8)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)  # Tiny fc layer
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

alexnet_10 = lambda : alexnet(num_classes=10)

convnext_tiny10 = lambda: models.convnext_tiny(num_classes = 10)

convnext_small10 = lambda: models.convnext_small(num_classes = 10)

convnext_base10 = lambda: models.convnext_base(num_classes = 10)

convnext_large10 = lambda: models.convnext_large(num_classes = 10)

densenet121_10 = lambda: models.densenet121(num_classes = 10)
densenet169_10 = lambda: models.densenet169(num_classes = 10)
densenet201_10 = lambda: models.densenet201(num_classes = 10)

efficientnet_b0_10 = lambda: models.efficientnet_b0(num_classes = 10)
efficientnet_b3_10 = lambda: models.efficientnet_b3(num_classes = 10)
efficientnet_b5_10 = lambda: models.efficientnet_b5(num_classes = 10)
efficientnet_b7_10 = lambda: models.efficientnet_b7(num_classes = 10)