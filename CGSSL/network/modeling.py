from .utils import IntermediateLayerGetter
from ._unet import UNet
from ._deeplab import DeepLabHeadV3Plus
from .utils import SimpleSegmentationModel, SimpleSegmentationModelDual
from .backbone import resnet
from ._pspnet import PSPHead
from ._segnet import SegNet
from torch import nn
import torch
import torchvision
import re




def unet(in_number, num_classes, output_stride=8, pretrained_backbone=False):
    model = UNet(in_number, num_classes)
    return model

def deeplabv3plus(in_number, num_classes, output_stride=16, pretrained_backbone=False):
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    backbone = resnet.__dict__['resnet50'](pretrained=pretrained_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
    backbone.conv1 = nn.Conv2d(in_number, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    inplanes = 2048
    low_level_planes = 256
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = SimpleSegmentationModel(backbone, classifier)
    return model


def pspnet(in_number, num_classes, output_stride=16, pretrained_backbone=False):
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    backbone = resnet.__dict__['resnet50'](pretrained=pretrained_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
    backbone.conv1 = nn.Conv2d(in_number, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    inplanes = 2048
    low_level_planes = 256
    return_layers = {'layer4': 'out'}
    classifier = PSPHead(inplanes, num_classes, nn.BatchNorm2d)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = SimpleSegmentationModel(backbone, classifier)
    return model

def segnet(in_number, num_classes):
    model = SegNet(input_channels=in_number, output_channels=num_classes, pretrained_backbone=False)
    return model