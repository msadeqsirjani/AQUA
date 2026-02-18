"""
MobileNetV2 for CIFAR-10 (Sandler et al., 2018).

CIFAR variant: stride-1 first conv (no aggressive downsampling on 32x32).
Uses inverted residual blocks with linear bottleneck.
"""

import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        self.use_res = stride == 1 and in_ch == out_ch
        hidden = int(round(in_ch * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_ch, hidden, kernel_size=1))
        layers += [
            ConvBNReLU(hidden, hidden, stride=stride, groups=hidden),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Model(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super().__init__()
        # (expand_ratio, out_channels, n_repeats, stride)
        settings = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],   # stride 1 for CIFAR (was 2 for ImageNet)
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        in_ch = int(32 * width_mult)
        last_ch = int(1280 * width_mult) if width_mult > 1.0 else 1280

        features = [ConvBNReLU(3, in_ch, stride=1)]  # stride 1 for CIFAR
        for t, c, n, s in settings:
            out_ch = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(in_ch, out_ch, stride, t))
                in_ch = out_ch
        features.append(ConvBNReLU(in_ch, last_ch, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_ch, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def MobileNetV2(num_classes=10):
    return MobileNetV2Model(num_classes=num_classes)
