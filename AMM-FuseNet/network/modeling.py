from .utils import IntermediateLayerGetter

from ._dual_parasingle import dualParasingleHead

from .utils import SimpleSegmentationModel, SimpleSegmentationModelDual
from .backbone import resnet
from .backbone import eca_resnet
from torch import nn

import torch
import torchvision
import re


def _segm_dual_resnet(dataset, name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    if backbone_name == 'dual_parasingle_pretrained_nopretrained':
        backbone1 = resnet.__dict__["resnet50"](
            pretrained=True,
            replace_stride_with_dilation=replace_stride_with_dilation)
        backbone2 = eca_resnet.__dict__["eca_resnet50"](
            pretrained=False,
            num_classes=num_classes)
    if backbone_name == 'resnet50':
        backbone1 = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation)
        backbone2 = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation)
    elif backbone_name == 'eca_resnet50':
        backbone1 = eca_resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            num_classes=num_classes)
        backbone2 = eca_resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            num_classes=num_classes)


    if dataset == 'hunan':
        backbone1.conv1 = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone2.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif dataset == 'hunan2':
        backbone1.conv1 = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    elif dataset == 'potsdam':
        backbone1.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif dataset == 'dfc20':
        backbone1.conv1 = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone2.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    else:
        raise RuntimeError ("dataset error")
    inplanes = 2048
    low_level_planes = 256

    if name == 'dual_parasingle':
        return_layers = {'layer4': 'out', 'layer1': 'low_level', 'layer2': 'block2', 'layer3': 'block3'}
        classifier = dualParasingleHead(inplanes, low_level_planes, num_classes, aspp_dilate)

    else:
        raise RuntimeError ("network error")
    backbone1 = IntermediateLayerGetter(backbone1, return_layers=return_layers)
    backbone2 = IntermediateLayerGetter(backbone2, return_layers=return_layers)
    model = SimpleSegmentationModelDual(backbone1, backbone2, classifier)
    return model

def dual_parasingle_nopretrained(dataset, num_classes, output_stride=8, pretrained_backbone=False):
    model = _segm_dual_resnet(dataset, 'dual_parasingle', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    return model

def dual_parasingle_pretrained(dataset, num_classes, output_stride=8, pretrained_backbone=True):
    model = _segm_dual_resnet(dataset, 'dual_parasingle', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    return model

