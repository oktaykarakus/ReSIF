from .utils import IntermediateLayerGetter

from ._dual_parasingle import ThreePlusOneHead

from .utils import SegmentationModel3plus1
from .backbone import resnet
from torch import nn


def _segm_3_resnet_plus_1(dataset, name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    if backbone_name == 'resnet50':
        backbone1 = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation)
        backbone2 = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation)
        backbone3 = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation)
    else:
        raise NotImplementedError(f'specified backbone \'{backbone_name}\' not implemented')

    if dataset == 'passau':
        backbone1.conv1 = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # sentinel-2 data
        backbone2.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # planet data
        backbone3.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # dem data
    else:
        raise NotImplementedError(f'specified dataset \'{dataset}\' not implemented')

    inplanes = 2048
    low_level_planes = 256

    if name == 'tri_parasingle':
        return_layers = {'layer4': 'out', 'layer1': 'low_level', 'layer2': 'block2', 'layer3': 'block3'}
        # TODO: adjust classifier/regressor
        classifier = ThreePlusOneHead(inplanes, low_level_planes, num_classes, aspp_dilate)

    else:
        raise RuntimeError("network error")
    backbone1 = IntermediateLayerGetter(backbone1, return_layers=return_layers)
    backbone2 = IntermediateLayerGetter(backbone2, return_layers=return_layers)
    backbone3 = IntermediateLayerGetter(backbone3, return_layers=return_layers)
    model = SegmentationModel3plus1(backbone1, backbone2, backbone3, classifier)
    return model


def threeplusone_pretrained(dataset, num_classes, output_stride=8, pretrained_backbone=True):
    model = _segm_3_resnet_plus_1(dataset, 'tri_parasingle', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    return model
