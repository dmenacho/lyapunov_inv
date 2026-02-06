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
    

class SmallVGG(nn.Module):
    """Compact VGG-style network for MNIST"""
    
    def __init__(self, num_classes=10, dropout=0.3):
        super(SmallVGG, self).__init__()
        
        # Feature extractor with reduced channels
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
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # Padding to get 3x3 from 7x7
            nn.Dropout2d(dropout),
        )
        
        # Classifier with reduced parameters
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
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
            
            # 7x7 -> 4x4 (with padding)
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
        
        # Feature extractor with reduced channels
        self.features = nn.Sequential(
            # Layer 1: 28x28 -> 27x27 -> 13x13
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Adjusted for MNIST
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
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # 7->4 (with padding)
        )
        
        # Classifier with reduced parameters
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, 256),  # Adjusted for 4x4 output
            nn.ReLU(inplace=True),
            
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, num_classes),
        )
        
        # Initialize weights
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
        
        # Reduced architecture
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
        
        # Adaptive pooling to handle variable sizes
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
            nn.SiLU(inplace=True),  # Swish activation
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
        
        # Expansion phase (only if expand_ratio > 1)
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand = nn.Identity()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride, kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze and excitation
        if se_ratio is not None and se_ratio > 0:
            self.se = SEBlock(expanded_channels, reduction=int(1/se_ratio))
        else:
            self.se = nn.Identity()
        
        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        # Expansion
        out = self.expand(x)
        
        # Depthwise convolution
        out = self.depthwise(out)
        
        # SE block
        out = self.se(out)
        
        # Pointwise convolution
        out = self.pointwise(out)
        
        # Skip connection
        if self.use_residual:
            out = out + identity
        
        return out


# Simplified MBConv Block (no expansion/SE for memory efficiency)
class SimpleMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(SimpleMBConvBlock, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        self.block = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                     kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            
            # Pointwise
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
        
        # Stem (adjusted for MNIST 28x28)
        stem_channels = self._round_channels(16, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True)
        )
        
        # Build blocks
        blocks = []
        
        # Stage 1: 28x28 -> 14x14
        in_channels = stem_channels
        out_channels = self._round_channels(16, width_mult)
        blocks.append(MBConvBlock(in_channels, out_channels, stride=1, expand_ratio=1))
        
        # Stage 2: 14x14 -> 7x7
        in_channels = out_channels
        out_channels = self._round_channels(24, width_mult)
        blocks.append(MBConvBlock(in_channels, out_channels, stride=2, expand_ratio=6))
        
        # Stage 3: 7x7 -> 4x4 (with padding adjustment)
        in_channels = out_channels
        out_channels = self._round_channels(32, width_mult)
        blocks.append(MBConvBlock(in_channels, out_channels, stride=2, expand_ratio=6))
        
        # Stage 4: 4x4 -> 4x4
        in_channels = out_channels
        out_channels = self._round_channels(64, width_mult)
        blocks.append(MBConvBlock(in_channels, out_channels, stride=1, expand_ratio=6))
        
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        final_channels = self._round_channels(128, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU(inplace=True)
        )
        
        # Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(final_channels, num_classes)
        
        # Initialize weights
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
            nn.Conv2d(1, 8, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)  # Using ReLU for simplicity
        )
        
        # Simplified blocks
        self.blocks = nn.Sequential(
            # Stage 1: 28x28 -> 14x14
            SimpleMBConvBlock(8, 16, stride=2),
            
            # Stage 2: 14x14 -> 7x7
            SimpleMBConvBlock(16, 24, stride=2),
            
            # Stage 3: 7x7 -> 4x4
            SimpleMBConvBlock(24, 32, stride=2),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(32, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(48, num_classes)
    
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
        
        # Configuration for different versions
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
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, config['channels'][0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(config['channels'][0]),
            nn.SiLU(inplace=True)
        )
        
        # Build blocks
        blocks = []
        in_channels = config['channels'][0]
        
        for i in range(4):
            out_channels = config['channels'][i+1]
            stride = config['strides'][i]
            expand_ratio = config['expand_ratios'][i]
            
            # Use simplified block for early layers to save memory
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
        
        # Scale channels by width multiplier
        channels = [max(4, int(c * width_mult)) for c in channels]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )
        
        # Build blocks
        blocks = []
        in_channels = channels[0]
        
        for i in range(len(strides)):
            out_channels = channels[i+1]
            stride = strides[i]
            expand_ratio = expand_ratios[i] if i < len(expand_ratios) else 1
            
            if use_se and i >= 1:  # Add SE to later blocks
                se_ratio = 0.25
            else:
                se_ratio = None
            
            blocks.append(MBConvBlock(in_channels, out_channels, 
                                     stride=stride, expand_ratio=expand_ratio,
                                     se_ratio=se_ratio))
            in_channels = out_channels
        
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        final_channels = max(32, in_channels * 2)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            nn.SiLU(inplace=True)
        )
        
        # Pooling and classifier
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
        # x: (B, C, H, W)
        # LayerNorm expects (B, H, W, C) or (B, C)
        # We'll permute to (B, H, W, C), apply norm, then permute back
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
        # Compute mean and variance per channel
        var, mean = torch.var_mean(x, dim=(1, 2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


# ConvNeXt Block
class ConvNeXtBlock(nn.Module):
    """Original ConvNeXt block with depthwise 7x7 conv"""
    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        
        # Depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm
        self.norm = LayerNorm2d(dim)
        
        # Pointwise 1x1 convs (equivalent to MLP)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        
        # Layer scale
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim, 1, 1), 
                                 requires_grad=True) if layer_scale_init > 0 else None
        
        # Drop path (stochastic depth)
        self.drop_path = nn.Identity()  # Simplified for MNIST
        
    def forward(self, x):
        input = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # LayerNorm + MLP
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm.norm(x)  # Directly use the LayerNorm
        x = x.permute(0, 3, 1, 2)  # Back to (B, C, H, W)
        
        # MLP
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer scale
        if self.gamma is not None:
            x = x * self.gamma
        
        # Residual connection
        x = input + self.drop_path(x)
        return x


# Simplified ConvNeXt Block (smaller kernel, no layer scale)
class SimpleConvNeXtBlock(nn.Module):
    """Simplified block with 5x5 kernel and reduced expansion"""
    def __init__(self, dim, kernel_size=5, expansion=2):
        super().__init__()
        
        self.block = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.GELU(),
            
            # Channel mixing with smaller expansion
            nn.Conv2d(dim, dim * expansion, 1),
            nn.GELU(),
            nn.Conv2d(dim * expansion, dim, 1),
        )
        
        # Simplified LayerNorm
        self.norm = SimpleLayerNorm2d(dim)
        
    def forward(self, x):
        return x + self.block(self.norm(x))


# Micro ConvNeXt Block for extreme memory constraints
class MicroConvNeXtBlock(nn.Module):
    """Tiny block with 3x3 kernel and minimal operations"""
    def __init__(self, dim):
        super().__init__()
        
        self.block = nn.Sequential(
            # Grouped convolution (depthwise)
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            
            # Simple channel mixing
            nn.Conv2d(dim, dim, 1),
        )
        
    def forward(self, x):
        return x + self.block(x)


# Downsampling layer
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
        
        # Stem: 28x28 -> 14x14
        self.stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=4, stride=4),  # Patchify
            SimpleLayerNorm2d(dims[0])
        )
        
        # Stages
        self.stages = nn.ModuleList()
        curr_dim = dims[0]
        
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            # First block in each stage (except first) can downsample
            if i > 0:
                # Downsample
                downsample = Downsample(curr_dim, dim)
                curr_dim = dim
                stage = [downsample]
            else:
                stage = []
            
            # Add blocks
            for _ in range(depth):
                stage.append(ConvNeXtBlock(
                    curr_dim, 
                    drop_path=drop_path_rate,
                    layer_scale_init=layer_scale_init
                ))
            
            self.stages.append(nn.Sequential(*stage))
        
        # Head
        self.norm = SimpleLayerNorm2d(curr_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(curr_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
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
        # Stem
        x = self.stem(x)
        
        # Stages
        for stage in self.stages:
            x = stage(x)
        
        # Head
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class ConvNeXtMicro(nn.Module):
    """Micro ConvNeXt for MNIST (~30K params)"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Dimensions for MNIST
        dims = [16, 32, 64]
        depths = [1, 2, 2]  # Fewer blocks
        
        # Stem: aggressive downsampling for MNIST
        self.stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=3, stride=2, padding=1),  # 28->14
            SimpleLayerNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=2, padding=1),  # 14->7
        )
        
        # Stages with simplified blocks
        self.stage1 = self._make_stage(dims[0], dims[0], depths[0])
        self.stage2 = self._make_stage(dims[0], dims[1], depths[1])
        self.stage3 = self._make_stage(dims[1], dims[2], depths[2])
        
        # Head
        self.norm = SimpleLayerNorm2d(dims[2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(dims[2], 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )
    
    def _make_stage(self, in_dim, out_dim, num_blocks):
        layers = []
        
        # Downsample if dimensions change
        if in_dim != out_dim:
            layers.append(
                nn.Sequential(
                    SimpleLayerNorm2d(in_dim),
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
                )
            )
            in_dim = out_dim
        
        # Add blocks
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
        
        # Even smaller dimensions
        self.model = nn.Sequential(
            # Stem: 28x28 -> 14x14 -> 7x7
            nn.Conv2d(1, 8, 3, stride=2, padding=1),  # 28->14
            nn.GELU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),  # 14->7
            nn.GELU(),
            
            # Stage 1: 7x7
            MicroConvNeXtBlock(8),
            nn.GELU(),
            
            # Downsample: 7x7 -> 4x4
            nn.Conv2d(8, 16, 2, stride=2),
            nn.GELU(),
            
            # Stage 2: 4x4
            MicroConvNeXtBlock(16),
            nn.GELU(),
            
            # Downsample: 4x4 -> 2x2
            nn.Conv2d(16, 32, 2, stride=2),
            nn.GELU(),
            
            # Final block
            MicroConvNeXtBlock(32),
            nn.GELU(),
            
            # Head
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
        
        # Variant configurations
        configs = {
            'pico':   {'dims': [8, 16, 32],  'depths': [1, 1, 1], 'expansion': 2},
            'nano':   {'dims': [16, 32, 64], 'depths': [1, 2, 2], 'expansion': 2},
            'micro':  {'dims': [32, 64, 128], 'depths': [2, 2, 3], 'expansion': 4},
            'tiny':   {'dims': [48, 96, 192], 'depths': [2, 3, 4], 'expansion': 4},
            'small':  {'dims': [64, 128, 256], 'depths': [3, 4, 5], 'expansion': 4},
        }
        
        cfg = configs[variant]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, cfg['dims'][0], 4, stride=4),  # Patchify: 28x28 -> 7x7
            SimpleLayerNorm2d(cfg['dims'][0])
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        current_dim = cfg['dims'][0]
        
        for i in range(3):
            # Downsample if not first stage
            if i > 0:
                downsample = nn.Sequential(
                    SimpleLayerNorm2d(current_dim),
                    nn.Conv2d(current_dim, cfg['dims'][i], 2, stride=2)
                )
                self.stages.append(downsample)
                current_dim = cfg['dims'][i]
            
            # Add blocks for this stage
            for _ in range(cfg['depths'][i]):
                self.stages.append(
                    SimpleConvNeXtBlock(current_dim, expansion=cfg['expansion'])
                )
        
        # Head
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
        
        # Depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
        # Norm + MLP
        self.norm = SimpleLayerNorm2d(dim)
        
        # MLP with smaller expansion for memory efficiency
        self.pwconv1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        
        # Global Response Normalization
        self.grn = GRN(dim * expansion)
        
        self.pwconv2 = nn.Linear(dim * expansion, dim)
    
    def forward(self, x):
        input = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Permute for LayerNorm and linear layers
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm.norm(x) if hasattr(self.norm, 'norm') else self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # MLP with GRN
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        
        # Permute back and residual
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# ------------------------------------------------------------
# MobileNetV1 Components
# ------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution for MobileNetV1"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                  stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu6(self.bn1(self.depthwise(x)))
        x = F.relu6(self.bn2(self.pointwise(x)))
        return x


class MobileNetV1Tiny(nn.Module):
    """Tiny MobileNetV1 for MNIST"""
    
    def __init__(self, num_classes=10, width_multiplier=0.25):
        super().__init__()
        
        # Adjust channels with width multiplier
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        
        # Build layers
        layers = []
        
        # First conv layer (adjusted for MNIST)
        input_channel = int(8 * width_multiplier)
        layers.append(conv_bn(1, input_channel, 2))  # 28x28 -> 14x14
        
        # Depthwise separable conv blocks
        cfg = [
            # t, c, n, s (expansion, output channels, repeats, stride)
            [1, 16, 1, 1],
            [1, 32, 2, 2],
            [1, 64, 2, 2],
            [1, 128, 2, 1],
        ]
        
        for t, c, n, s in cfg:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(DepthwiseSeparableConv(input_channel, output_channel, stride))
                input_channel = output_channel
        
        # Final layers
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(input_channel, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ------------------------------------------------------------
# MobileNetV2 Components
# ------------------------------------------------------------

class InvertedResidual(nn.Module):
    """Inverted Residual Block for MobileNetV2"""
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                               groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # Pointwise projection
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Tiny(nn.Module):
    """Tiny MobileNetV2 for MNIST"""
    
    def __init__(self, num_classes=10, width_mult=0.25):
        super().__init__()
        
        # Config for tiny version
        inverted_residual_setting = [
            # t, c, n, s (expansion, output channels, repeats, stride)
            [1, 8, 1, 1],    # 14x14
            [3, 16, 2, 2],   # 7x7
            [3, 32, 3, 2],   # 4x4
            [3, 64, 1, 1],   # 4x4
        ]
        
        # Building first layer
        input_channel = int(8 * width_mult)
        last_channel = int(128 * width_mult)
        
        features = [nn.Sequential(
            nn.Conv2d(1, input_channel, 3, 2, 1, bias=False),  # 28x28 -> 14x14
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]
        
        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, 
                                                stride, expand_ratio=t))
                input_channel = output_channel
        
        # Building last layers
        features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ))
        
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ------------------------------------------------------------
# MobileNetV3 Components
# ------------------------------------------------------------

class HSwish(nn.Module):
    """Hard Swish activation"""
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation"""
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6


class SqueezeExcitation(nn.Module):
    """Squeeze and Excitation block"""
    def __init__(self, channel, reduction=4):
        super().__init__()
        reduced_channels = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channel, bias=False),
            HardSigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MobileNetV3Block(nn.Module):
    """MobileNetV3 building block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 use_se, activation, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(round(in_channels * expand_ratio))
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(activation())
        
        # Depthwise convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                               kernel_size//2, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(activation())
        
        # Squeeze and excitation
        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))
        
        # Projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileNetV3Tiny(nn.Module):
    """Tiny MobileNetV3 for MNIST"""
    
    def __init__(self, num_classes=10, mode='small', width_mult=0.25):
        super().__init__()
        
        # Configuration for tiny version
        if mode == 'small':
            # Small config for MNIST
            cfgs = [
                # k, exp, c,  se,     nl,  s
                [3,  1,   8,  False, 'RE', 2],  # 28x28 -> 14x14
                [3,  4,   16, False, 'RE', 2],  # 14x14 -> 7x7
                [3,  4,   24, False, 'RE', 2],  # 7x7 -> 4x4
                [3,  3,   32, True,  'HS', 1],  # 4x4
            ]
            last_channel = 64
        else:  # large
            cfgs = [
                [3,  1,   12, False, 'RE', 2],
                [3,  4,   24, False, 'RE', 2],
                [3,  3,   32, False, 'RE', 2],
                [5,  3,   48, True,  'HS', 1],
            ]
            last_channel = 96
        
        # Scale channels
        def scale_channels(channels):
            return int(channels * width_mult)
        
        # Building first layer
        input_channel = scale_channels(8)
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            HSwish() if mode == 'large' else nn.ReLU6(inplace=True)
        )
        
        # Building blocks
        blocks = []
        for k, exp, c, se, nl, s in cfgs:
            output_channel = scale_channels(c)
            exp_channel = input_channel * exp
            
            activation = HSwish if nl == 'HS' else nn.ReLU6
            
            blocks.append(MobileNetV3Block(
                input_channel, output_channel, k, s, 
                se, activation, exp
            ))
            input_channel = output_channel
        
        self.blocks = nn.Sequential(*blocks)
        
        # Building last layers
        last_conv_channels = scale_channels(last_channel)
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, last_conv_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_conv_channels),
            HSwish() if mode == 'large' else nn.ReLU6(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, scale_channels(32)),
            HSwish() if mode == 'large' else nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(scale_channels(32), num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.first_conv(x)      # 28x28 -> 14x14
        x = self.blocks(x)          # 14x14 -> 4x4
        x = self.last_conv(x)       # Feature refinement
        x = self.avgpool(x)         # Global pooling
        x = x.view(x.size(0), -1)   # Flatten
        x = self.classifier(x)      # Classify
        return x


# ------------------------------------------------------------
# Extremely Tiny MobileNet Variants
# ------------------------------------------------------------

class MobileNetMicro(nn.Module):
    """Micro MobileNet for extreme memory constraints"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Stem: 28x28 -> 14x14
            nn.Conv2d(1, 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU6(inplace=True),
            
            # Depthwise separable conv 1: 14x14 -> 7x7
            nn.Conv2d(4, 4, 3, stride=2, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU6(inplace=True),
            nn.Conv2d(4, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            
            # Depthwise separable conv 2: 7x7 -> 4x4
            nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            nn.Conv2d(8, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            
            # Depthwise separable conv 3: 4x4
            nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MobileNetNano(nn.Module):
    """Nano MobileNet - even smaller"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.model = nn.Sequential(
            # Stem
            nn.Conv2d(1, 4, 3, stride=2, padding=1),  # 28->14
            nn.ReLU6(),
            
            # Block 1
            nn.Conv2d(4, 4, 3, stride=2, padding=1, groups=4),  # 14->7
            nn.ReLU6(),
            nn.Conv2d(4, 8, 1),
            nn.ReLU6(),
            
            # Block 2
            nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=8),  # 7->4
            nn.ReLU6(),
            nn.Conv2d(8, 16, 1),
            nn.ReLU6(),
            
            # Global pooling and classification
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

