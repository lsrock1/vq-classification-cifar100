"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from .vq import Quantize

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, vq, student, num_class=100):
        super().__init__()
        self.features = features
        self.has_vq = vq
        self.student = student
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096), # modified because of teacher net
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
        if self.has_vq:
            self.vq = Quantize(2048, 1024, on_training=not student)
            self.quantize_conv_t = nn.Conv2d(512, 512, 1)

    def forward(self, x):
        output = self.features(x)
        if self.has_vq:
            output = self.quantize_conv_t(output)
            output = self.vq(output.permute(0, 2, 3, 1))
            output, diff, _ = output
            output = output.permute(0, 3, 1, 2)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        if self.training:
            return output, diff
        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(vq, student):
    return VGG(make_layers(cfg['A'], batch_norm=True), vq, student)

def vgg13_bn(vq, student):
    return VGG(make_layers(cfg['B'], batch_norm=True), vq, student)

def vgg16_bn(vq, student):
    return VGG(make_layers(cfg['D'], batch_norm=True), vq, student)

def vgg19_bn(vq, student):
    return VGG(make_layers(cfg['E'], batch_norm=True), vq, student)


