import torch
from torch import nn
from torch.nn import functional as F
from .eca_module import eca_layer


class ThreePlusOneHead(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(ThreePlusOneHead, self).__init__()
        self.project_image = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.project_dsm = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.dense_aspp_image = DenseASPP(low_level_channels)
        self.dense_aspp_dsm = DenseASPP(low_level_channels)

        # actual classifier implementation
        # TODO: make this make sense
        self.classifier = nn.Sequential(
            nn.Conv2d(448 * 3 + 1, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
            nn.Sigmoid()
        )
        self._init_weight()

        self.project_block2_image = nn.Sequential(
            nn.Conv2d(low_level_channels * 2, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.project_block3_image = nn.Sequential(
            nn.Conv2d(low_level_channels * 4, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.project_block4_image = nn.Sequential(
            nn.Conv2d(low_level_channels * 8, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),)

        self.project_block2_dsm = nn.Sequential(
            nn.Conv2d(low_level_channels * 2, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.project_block3_dsm = nn.Sequential(
            nn.Conv2d(low_level_channels * 4, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.project_block4_dsm = nn.Sequential(
            nn.Conv2d(low_level_channels * 8, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True), )

        self.eca_net_1_image = eca_layer(low_level_channels)
        self.eca_net_2_image = eca_layer(low_level_channels * 2)
        self.eca_net_3_image = eca_layer(low_level_channels * 4)
        self.eca_net_4_image = eca_layer(low_level_channels * 8)

        self.eca_net_1_dsm = eca_layer(low_level_channels)
        self.eca_net_2_dsm = eca_layer(low_level_channels * 2)
        self.eca_net_3_dsm = eca_layer(low_level_channels * 4)
        self.eca_net_4_dsm = eca_layer(low_level_channels * 8)

    def forward(self, features_imageA, features_imageB, features_dsm, wind):
        low_level_feature_imageA = self.project_image(features_imageA['low_level'])
        low_level_feature_imageA = self.eca_net_1_image(low_level_feature_imageA)

        low_level_feature_imageB = self.project_image(features_imageB['low_level'])
        low_level_feature_imageB = self.eca_net_1_image(low_level_feature_imageB)

        low_level_feature_dsm = self.project_dsm(features_dsm['low_level'])
        low_level_feature_dsm = self.eca_net_1_dsm(low_level_feature_dsm)

        block2_imageA = self.project_block2_image(features_imageA['block2'])
        block2_imageA = self.eca_net_2_image(block2_imageA)
        block2_imageA = F.interpolate(block2_imageA, size=low_level_feature_imageA.shape[2:], mode='bilinear', align_corners=False)

        block2_imageB = self.project_block2_image(features_imageB['block2'])
        block2_imageB = self.eca_net_2_image(block2_imageB)
        block2_imageB = F.interpolate(block2_imageB, size=low_level_feature_imageB.shape[2:], mode='bilinear', align_corners=False)

        block2_dsm = self.project_block2_dsm(features_dsm['block2'])
        block2_dsm = self.eca_net_2_dsm(block2_dsm)
        block2_dsm = F.interpolate(block2_dsm, size=low_level_feature_dsm.shape[2:], mode='bilinear', align_corners=False)

        block3_imageA = self.project_block3_image(features_imageA['block3'])
        block3_imageA = self.eca_net_3_image(block3_imageA)

        block3_imageB = self.project_block3_image(features_imageB['block3'])
        block3_imageB = self.eca_net_3_image(block3_imageB)

        block3_dsm = self.project_block3_dsm(features_dsm['block3'])
        block3_dsm = self.eca_net_3_dsm(block3_dsm)

        block4_imageA = self.project_block4_image(features_imageA['out'])
        block4_imageA = self.eca_net_4_image(block4_imageA)

        block4_imageB = self.project_block4_image(features_imageB['out'])
        block4_imageB = self.eca_net_4_image(block4_imageB)

        block4_dsm = self.project_block4_dsm(features_dsm['out'])
        block4_dsm = self.eca_net_4_dsm(block4_dsm)

        output_feature_imageA_aspp = self.dense_aspp_image(features_imageA['low_level'])
        output_feature_imageA = torch.cat((block3_imageA, block4_imageA), dim=1)
        output_feature_imageA = F.interpolate(output_feature_imageA, size=low_level_feature_imageA.shape[2:], mode='bilinear', align_corners=False)

        output_feature_imageB_aspp = self.dense_aspp_image(features_imageB['low_level'])
        output_feature_imageB = torch.cat((block3_imageB, block4_imageB), dim=1)
        output_feature_imageB = F.interpolate(output_feature_imageB, size=low_level_feature_imageB.shape[2:], mode='bilinear', align_corners=False)

        output_feature_dsm_aspp = self.dense_aspp_dsm(features_dsm['low_level'])
        output_feature_dsm = torch.cat((block3_dsm, block4_dsm), dim=1)
        output_feature_dsm = F.interpolate(output_feature_dsm, size=low_level_feature_dsm.shape[2:],
                                           mode='bilinear', align_corners=False)

        return self.classifier(torch.cat([wind, low_level_feature_imageA, block2_imageA, output_feature_imageA, output_feature_imageA_aspp, low_level_feature_imageB, block2_imageB, output_feature_imageB, output_feature_imageB_aspp, low_level_feature_dsm, block2_dsm, output_feature_dsm, output_feature_dsm_aspp], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            eca_layer(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class DenseASPP(nn.Module):
    def __init__(self, in_channels):
        super(DenseASPP, self).__init__()
        out_channels = 256
        atrous_rates = (3, 6, 12, 18)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        rate1, rate2, rate3, rate4 = tuple(atrous_rates)
        self.asppConv_1 = ASPPConv(in_channels, out_channels, rate1)
        self.asppConv_2 = ASPPConv(in_channels + out_channels * 1, out_channels, rate2)
        self.asppConv_3 = ASPPConv(in_channels + out_channels * 2, out_channels, rate3)
        self.asppConv_4 = ASPPConv(in_channels + out_channels * 3, out_channels, rate4)
        self.asppPooling = ASPPPooling(in_channels, out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels + in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)  # there is a dropout layer, good.

    def forward(self, x):
        feature = x
        ASPP_1 = self.asppConv_1(feature)
        feature = torch.cat((feature, ASPP_1), dim=1)
        ASPP_2 = self.asppConv_2(feature)
        feature = torch.cat((feature, ASPP_2), dim=1)
        ASPP_3 = self.asppConv_3(feature)
        feature = torch.cat((feature, ASPP_3), dim=1)
        ASPP_4 = self.asppConv_4(feature)
        pooling = self.asppPooling(x)
        res = torch.cat((x, ASPP_1, ASPP_2, ASPP_3, ASPP_4, pooling), dim=1)
        out = self.project(res)
        return out


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module
