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
        self.fc = nn.Linear(8, num_classes) 
    
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
        self.fc = nn.Linear(8, num_classes)  
    
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
    

class SmallVGG(nn.Module):
    """Compact VGG-style network for MNIST"""
    
    def __init__(self, num_classes=10, dropout=0.3):
        super(SmallVGG, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout/2),
            
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
            
            # Block 3: 7x7 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  
            nn.Dropout2d(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TinyVGG(nn.Module):
    """Even smaller version for memory-intensive experiments"""
    
    def __init__(self, num_classes=10):
        super(TinyVGG, self).__init__()
        
        self.features = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 14x14 -> 7x7
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 7x7 -> 4x4 
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MinimalVGG(nn.Module):
    """Minimal VGG for extreme memory constraints"""
    
    def __init__(self, num_classes=10):
        super(MinimalVGG, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class SmallAlexNet(nn.Module):
    """Compact AlexNet-style network for MNIST"""
    
    def __init__(self, num_classes=10, dropout=0.3):
        super(SmallAlexNet, self).__init__()

        self.features = nn.Sequential(
            # Layer 1: 28x28 -> 27x27 -> 13x13
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28->14
            
            # Layer 2: 14x14 -> 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14->7
            
            # Layer 3: 7x7 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4: 7x7 -> 7x7
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5: 7x7 -> 7x7 -> 3x3
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # 7->4 
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, 256), 
            nn.ReLU(inplace=True),
            
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TinyAlexNet(nn.Module):
    """Even smaller AlexNet for memory-intensive experiments"""
    
    def __init__(self, num_classes=10):
        super(TinyAlexNet, self).__init__()

        self.features = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 14x14 -> 7x7
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 7x7 -> 4x4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AlexNetLight(nn.Module):
    """Lightweight AlexNet with batch normalization"""
    
    def __init__(self, num_classes=10):
        super(AlexNetLight, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 5
            nn.Conv2d(64, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(48 * 3 * 3, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetLocalResponseNorm(nn.Module):
    """AlexNet with Local Response Normalization (LRN) like original"""
    
    def __init__(self, num_classes=10):
        super(AlexNetLocalResponseNorm, self).__init__()
        
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5
            nn.Conv2d(64, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(48 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.SiLU(inplace=True),  
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


# MBConv Block (Mobile Inverted Bottleneck with optional SE)
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, expand_ratio=1, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        expanded_channels = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand = nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride, kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )

        if se_ratio is not None and se_ratio > 0:
            self.se = SEBlock(expanded_channels, reduction=int(1/se_ratio))
        else:
            self.se = nn.Identity()
 
        self.pointwise = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x

        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.pointwise(out)

        if self.use_residual:
            out = out + identity
        
        return out


# Simplified MBConv Block (no expansion/SE for memory efficiency)
class SimpleMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(SimpleMBConvBlock, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        self.block = nn.Sequential(

            nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                     kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.use_residual:
            out = out + identity
        return out


class TinyEfficientNet(nn.Module):
    """Compact EfficientNet for MNIST"""
    
    def __init__(self, num_classes=10, width_mult=1.0, depth_mult=1.0, 
                 dropout_rate=0.2, stochastic_depth=False):
        super(TinyEfficientNet, self).__init__()

        stem_channels = self._round_channels(16, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True)
        )

        blocks = []
        in_channels = stem_channels
        out_channels = self._round_channels(16, width_mult)
        blocks.append(MBConvBlock(in_channels, out_channels, stride=1, expand_ratio=1))

        in_channels = out_channels
        out_channels = self._round_channels(24, width_mult)
        blocks.append(MBConvBlock(in_channels, out_channels, stride=2, expand_ratio=6))

        in_channels = out_channels
        out_channels = self._round_channels(32, width_mult)
        blocks.append(MBConvBlock(in_channels, out_channels, stride=2, expand_ratio=6))

        in_channels = out_channels
        out_channels = self._round_channels(64, width_mult)
        blocks.append(MBConvBlock(in_channels, out_channels, stride=1, expand_ratio=6))
        
        self.blocks = nn.Sequential(*blocks)

        final_channels = self._round_channels(128, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(final_channels, num_classes)
        self._initialize_weights()
    
    def _round_channels(self, channels, multiplier):
        if not multiplier:
            return channels
        return int(ceil(channels * multiplier))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class EfficientNetMicro(nn.Module):
    """Micro EfficientNet for extreme memory constraints"""
    
    def __init__(self, num_classes=10):
        super(EfficientNetMicro, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True) 
        )
        
        self.blocks = nn.Sequential(
            # 28x28 -> 14x14
            SimpleMBConvBlock(4, 8, stride=2),
            
            # 14x14 -> 7x7
            SimpleMBConvBlock(16, 16, stride=2),
            
            # 7x7 -> 4x4
            SimpleMBConvBlock(16, 24, stride=2),
        )

        self.head = nn.Sequential(
            nn.Conv2d(24, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class EfficientNetNano(nn.Module):
    """Nano EfficientNet - middle ground"""
    
    def __init__(self, num_classes=10, version='nano'):
        super(EfficientNetNano, self).__init__()
        
        configs = {
            'nano': {
                'channels': [8, 12, 16, 24, 32],
                'strides': [1, 2, 2, 2],
                'expand_ratios': [1, 3, 3, 3]
            },
            'pico': {
                'channels': [4, 8, 12, 16, 24],
                'strides': [1, 2, 2, 2],
                'expand_ratios': [1, 2, 2, 2]
            }
        }
        
        config = configs[version]

        self.stem = nn.Sequential(
            nn.Conv2d(1, config['channels'][0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(config['channels'][0]),
            nn.SiLU(inplace=True)
        )

        blocks = []
        in_channels = config['channels'][0]
        
        for i in range(4):
            out_channels = config['channels'][i+1]
            stride = config['strides'][i]
            expand_ratio = config['expand_ratios'][i]

            if i < 2:
                blocks.append(SimpleMBConvBlock(in_channels, out_channels, stride=stride))
            else:
                blocks.append(MBConvBlock(in_channels, out_channels, 
                                         stride=stride, expand_ratio=expand_ratio, 
                                         se_ratio=0.25))
            in_channels = out_channels
        
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.SiLU(inplace=True)
        )
        
        # Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(in_channels * 2, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class EfficientNetCustom(nn.Module):
    """Custom EfficientNet with configurable parameters"""
    
    def __init__(self, num_classes=10, 
                 channels=[8, 16, 24, 32],
                 strides=[2, 2, 2],
                 expand_ratios=[1, 6, 6],
                 use_se=True,
                 width_mult=0.5):
        super(EfficientNetCustom, self).__init__()

        channels = [max(4, int(c * width_mult)) for c in channels]

        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )

        blocks = []
        in_channels = channels[0]
        
        for i in range(len(strides)):
            out_channels = channels[i+1]
            stride = strides[i]
            expand_ratio = expand_ratios[i] if i < len(expand_ratios) else 1
            
            if use_se and i >= 1:
                se_ratio = 0.25
            else:
                se_ratio = None
            
            blocks.append(MBConvBlock(in_channels, out_channels, 
                                     stride=stride, expand_ratio=expand_ratio,
                                     se_ratio=se_ratio))
            in_channels = out_channels
        
        self.blocks = nn.Sequential(*blocks)

        final_channels = max(32, in_channels * 2)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(final_channels, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (2D spatial data)"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


# Simplified LayerNorm2d for efficiency
class SimpleLayerNorm2d(nn.Module):
    """Simplified LayerNorm2d for memory-constrained scenarios"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps
    
    def forward(self, x):
        var, mean = torch.var_mean(x, dim=(1, 2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


# ConvNeXt Block
class ConvNeXtBlock(nn.Module):
    """Original ConvNeXt block with depthwise 7x7 conv"""
    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim, 1, 1), 
                                 requires_grad=True) if layer_scale_init > 0 else None

        self.drop_path = nn.Identity()  
        
    def forward(self, x):
        input = x
        
        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm.norm(x) 
        x = x.permute(0, 3, 1, 2)  # Back to (B, C, H, W)
        
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = x * self.gamma

        x = input + self.drop_path(x)
        return x


# Simplified ConvNeXt Block (smaller kernel, no layer scale)
class SimpleConvNeXtBlock(nn.Module):
    """Simplified block with 5x5 kernel and reduced expansion"""
    def __init__(self, dim, kernel_size=5, expansion=2):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.GELU(),
            
            nn.Conv2d(dim, dim * expansion, 1),
            nn.GELU(),
            nn.Conv2d(dim * expansion, dim, 1),
        )
        
        self.norm = SimpleLayerNorm2d(dim)
        
    def forward(self, x):
        return x + self.block(self.norm(x))


# Micro ConvNeXt Block for extreme memory constraints
class MicroConvNeXtBlock(nn.Module):
    """Tiny block with 3x3 kernel and minimal operations"""
    def __init__(self, dim):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        
    def forward(self, x):
        return x + self.block(x)


class Downsample(nn.Module):
    """Downsampling layer for ConvNeXt"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            SimpleLayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.down(x)


class TinyConvNeXt(nn.Module):
    """Tiny ConvNeXt for MNIST (~100K params)"""
    
    def __init__(self, num_classes=10, depths=[2, 2, 4], dims=[32, 64, 128], 
                 drop_path_rate=0.0, layer_scale_init=1e-6):
        super().__init__()
        
        #  28x28 -> 14x14
        self.stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=4, stride=4),  
            SimpleLayerNorm2d(dims[0])
        )

        self.stages = nn.ModuleList()
        curr_dim = dims[0]
        
        for i, (depth, dim) in enumerate(zip(depths, dims)):

            if i > 0:
                downsample = Downsample(curr_dim, dim)
                curr_dim = dim
                stage = [downsample]
            else:
                stage = []

            for _ in range(depth):
                stage.append(ConvNeXtBlock(
                    curr_dim, 
                    drop_path=drop_path_rate,
                    layer_scale_init=layer_scale_init
                ))
            
            self.stages.append(nn.Sequential(*stage))

        self.norm = SimpleLayerNorm2d(curr_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(curr_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, SimpleLayerNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):

        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class ConvNeXtMicro(nn.Module):
    """Micro ConvNeXt for MNIST (~30K params)"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        dims = [16, 32, 64]
        depths = [1, 2, 2]

        self.stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=3, stride=2, padding=1),  # 28->14
            SimpleLayerNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=2, padding=1),  # 14->7
        )
        
        self.stage1 = self._make_stage(dims[0], dims[0], depths[0])
        self.stage2 = self._make_stage(dims[0], dims[1], depths[1])
        self.stage3 = self._make_stage(dims[1], dims[2], depths[2])
        
        self.norm = SimpleLayerNorm2d(dims[2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(dims[2], 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )
    
    def _make_stage(self, in_dim, out_dim, num_blocks):
        layers = []

        if in_dim != out_dim:
            layers.append(
                nn.Sequential(
                    SimpleLayerNorm2d(in_dim),
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
                )
            )
            in_dim = out_dim
        
        for _ in range(num_blocks):
            layers.append(SimpleConvNeXtBlock(in_dim, kernel_size=5, expansion=2))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)       # 28x28 -> 7x7
        x = self.stage1(x)     # 7x7
        x = self.stage2(x)     # 7x7 -> 4x4
        x = self.stage3(x)     # 4x4
        x = self.norm(x)
        x = self.avgpool(x)    # 4x4 -> 1x1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class ConvNeXtNano(nn.Module):
    """Nano ConvNeXt for extreme memory constraints (~10K params)"""
    
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = nn.Sequential(
            # Stem: 28x28 -> 14x14 -> 7x7
            nn.Conv2d(1, 8, 3, stride=2, padding=1),  # 28->14
            nn.GELU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),  # 14->7
            nn.GELU(),
            
            MicroConvNeXtBlock(8),
            nn.GELU(),

            nn.Conv2d(8, 16, 2, stride=2),
            nn.GELU(),

            MicroConvNeXtBlock(16),
            nn.GELU(),
            
            nn.Conv2d(16, 32, 2, stride=2),
            nn.GELU(),
            
            MicroConvNeXtBlock(32),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class ConvNeXtVariant(nn.Module):
    """Configurable ConvNeXt variant"""
    
    def __init__(self, num_classes=10, variant='tiny'):
        super().__init__()

        configs = {
            'pico':   {'dims': [8, 16, 32],  'depths': [1, 1, 1], 'expansion': 2},
            'nano':   {'dims': [16, 32, 64], 'depths': [1, 2, 2], 'expansion': 2},
            'micro':  {'dims': [32, 64, 128], 'depths': [2, 2, 3], 'expansion': 4},
            'tiny':   {'dims': [48, 96, 192], 'depths': [2, 3, 4], 'expansion': 4},
            'small':  {'dims': [64, 128, 256], 'depths': [3, 4, 5], 'expansion': 4},
        }
        
        cfg = configs[variant]

        self.stem = nn.Sequential(
            nn.Conv2d(1, cfg['dims'][0], 4, stride=4),  # Patchify: 28x28 -> 7x7
            SimpleLayerNorm2d(cfg['dims'][0])
        )

        self.stages = nn.ModuleList()
        current_dim = cfg['dims'][0]
        
        for i in range(3):
            if i > 0:
                downsample = nn.Sequential(
                    SimpleLayerNorm2d(current_dim),
                    nn.Conv2d(current_dim, cfg['dims'][i], 2, stride=2)
                )
                self.stages.append(downsample)
                current_dim = cfg['dims'][i]

            for _ in range(cfg['depths'][i]):
                self.stages.append(
                    SimpleConvNeXtBlock(current_dim, expansion=cfg['expansion'])
                )
        
        self.norm = SimpleLayerNorm2d(current_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(current_dim, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        for layer in self.stages:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


# Modern ConvNeXt V2 style block with GRN
class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 block with Global Response Normalization"""
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
        self.norm = SimpleLayerNorm2d(dim)
        
        # MLP with smaller expansion for memory efficiency
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()

        self.grn = GRN(dim * expansion)
        
        self.pwconv2 = nn.Linear(dim * expansion, dim)
    
    def forward(self, x):
        input = x
        
        x = self.dwconv(x)
        
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm.norm(x) if hasattr(self.norm, 'norm') else self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)
        return input + x


class GRN(nn.Module):
    """Global Response Normalization from ConvNeXt V2"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x):
        # x: (B, H, W, C)
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # (B, 1, 1, C)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)   # Normalize
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Tiny(nn.Module):
    """ConvNeXt V2 style for MNIST"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=4),  # 28x28 -> 7x7
            SimpleLayerNorm2d(32)
        )
        
        # Stages
        self.stage1 = nn.Sequential(
            ConvNeXtV2Block(32, expansion=2),
            ConvNeXtV2Block(32, expansion=2),
        )
        
        self.down1 = nn.Sequential(
            SimpleLayerNorm2d(32),
            nn.Conv2d(32, 64, 2, stride=2)  # 7x7 -> 4x4
        )
        
        self.stage2 = nn.Sequential(
            ConvNeXtV2Block(64, expansion=2),
            ConvNeXtV2Block(64, expansion=2),
        )
        
        # Head
        self.norm = SimpleLayerNorm2d(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.stem(x)       # 28->7
        x = self.stage1(x)     # 7
        x = self.down1(x)      # 7->4
        x = self.stage2(x)     # 4
        x = self.norm(x)
        x = self.avgpool(x)    # 4->1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


## UNET 
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class TinyUNet(nn.Module):
    """
    U-Net-ish encoder/decoder with skip connections, used for classification:
        feature_map -> GAP -> MLP -> logits
    Designed for MNIST: input (B,1,28,28)
    """
    def __init__(self, num_classes=10, base=4, mlp_hidden=32):
        super().__init__()

        self.enc1 = ConvBlock(1, base)           # 28x28
        self.pool1 = nn.MaxPool2d(2)             # 14x14

        self.enc2 = ConvBlock(base, base*2)      # 14x14
        self.pool2 = nn.MaxPool2d(2)             # 7x7

        self.bottleneck = ConvBlock(base*2, base*4)  # 7x7

        self.up2 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2, bias=False)  
        self.dec2 = ConvBlock(base*2 + base*2, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2, bias=False)   
        self.dec1 = ConvBlock(base + base, base)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Linear(base, num_classes)

    def forward(self, x):

        x1 = self.enc1(x)          
        x2 = self.enc2(self.pool1(x1))  

        xb = self.bottleneck(self.pool2(x2))   

        y2 = self.up2(xb)          
        y2 = self.dec2(torch.cat([y2, x2], dim=1))

        y1 = self.up1(y2)                  
        y1 = self.dec1(torch.cat([y1, x1], dim=1))

        feat = self.gap(y1).flatten(1)         
        logits = self.head(feat)  
        return logits