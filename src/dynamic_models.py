import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from  torchvision.models import alexnet
from  torchvision import models
import numpy as np

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



class LeNet(nn.Module):

    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)