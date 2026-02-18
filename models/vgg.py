"""
VGG for CIFAR-10 / CIFAR-100 / Tiny-ImageNet.

CIFAR variant with BatchNorm after every conv and a compact classifier head
(no 4096-unit hidden layers).  ``AdaptiveAvgPool2d`` before the classifier
ensures the model works with any input resolution (32x32, 64x64, etc.).
"""

import torch.nn as nn


_cfgs = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
              512, 512, 512, "M", 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, cfg_name, num_classes=10):
        super().__init__()
        self.features = self._make_layers(_cfgs[cfg_name])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [
                    nn.Conv2d(in_channels, v, 3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def VGG11(num_classes=10):
    return VGG("vgg11", num_classes=num_classes)


def VGG16(num_classes=10):
    return VGG("vgg16", num_classes=num_classes)
