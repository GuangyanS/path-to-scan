import copy
from collections import OrderedDict
from functools import partial

import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, stride, groups,
                 norm_layer, act, conv_layer=nn.Conv2d):
        super(ConvBNAct, self).__init__(
            conv_layer(
                in_channel, out_channel, kernel_size, stride=stride,
                padding=(kernel_size - 1) // 2, groups=groups, bias=False),
            norm_layer(out_channel),
            act(),
        )


class SEUnit(nn.Module):
    def __init__(self, in_channel, reduction_ratio=4,
                 act1=partial(nn.SiLU, inplace=True), act2=nn.Sigmoid):
        super(SEUnit, self).__init__()
        hidden_dim = in_channel // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channel, hidden_dim, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, in_channel, 1, bias=True)
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.fc1(x_se)
        x_se = self.act1(x_se)
        x_se = self.fc2(x_se)
        return x * self.act2(x_se)


class StochasticDepth(nn.Module):
    def __init__(self, prob, mode):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == 'row' else [1]
        mask = torch.empty(shape, device=x.device).bernoulli_(self.survival)
        return x * mask.div_(self.survival)


class MBConvConfig:
    def __init__(self, expand_ratio, kernel, stride, in_ch, out_ch, layers,
                 use_se, fused, act=nn.SiLU, norm_layer=nn.BatchNorm2d):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_layers = layers
        self.use_se = use_se
        self.fused = fused
        self.act = act
        self.norm_layer = norm_layer

    @staticmethod
    def adjust_channels(channel, factor, divisible=8):
        new_channel = channel * factor
        adjusted = max(
            divisible,
            (int(new_channel + divisible / 2) // divisible) * divisible)
        if adjusted < 0.9 * new_channel:
            adjusted += divisible
        return adjusted


class MBConv(nn.Module):
    def __init__(self, config, sd_prob=0.0):
        super(MBConv, self).__init__()
        inter_channel = config.adjust_channels(
            config.in_ch, config.expand_ratio)
        block = []

        if config.expand_ratio == 1:
            block.append((
                'fused',
                ConvBNAct(
                    config.in_ch, inter_channel, config.kernel, config.stride,
                    1, config.norm_layer, config.act)))
        elif config.fused:
            block.append((
                'fused',
                ConvBNAct(
                    config.in_ch, inter_channel, config.kernel, config.stride,
                    1, config.norm_layer, config.act)))
            block.append((
                'fused_point_wise',
                ConvBNAct(
                    inter_channel, config.out_ch, 1, 1, 1,
                    config.norm_layer, nn.Identity)))
        else:
            block.append((
                'linear_bottleneck',
                ConvBNAct(
                    config.in_ch, inter_channel, 1, 1, 1,
                    config.norm_layer, config.act)))
            block.append((
                'depth_wise',
                ConvBNAct(
                    inter_channel, inter_channel, config.kernel,
                    config.stride, inter_channel, config.norm_layer,
                    config.act)))
            block.append(('se', SEUnit(inter_channel, 4 * config.expand_ratio)))
            block.append((
                'point_wise',
                ConvBNAct(
                    inter_channel, config.out_ch, 1, 1, 1,
                    config.norm_layer, nn.Identity)))

        self.block = nn.Sequential(OrderedDict(block))
        self.use_skip_connection = (
            config.stride == 1 and config.in_ch == config.out_ch)
        self.stochastic_path = StochasticDepth(sd_prob, 'row')

    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out


class EfficientNetV2(nn.Module):
    def __init__(self, layer_infos, out_channels=1280, nclass=100,
                 in_channels=4, dropout=0.1, stochastic_depth=0.2,
                 block=MBConv, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super(EfficientNetV2, self).__init__()
        self.in_channel = layer_infos[0].in_ch
        self.final_stage_channel = layer_infos[-1].out_ch
        self.out_channels = out_channels
        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.stem = ConvBNAct(
            in_channels, self.in_channel, 3, 2, 1, norm_layer, act_layer)
        self.blocks = nn.Sequential(*self.make_stages(layer_infos, block))
        self.head = nn.Sequential(OrderedDict([
            ('bottleneck', ConvBNAct(
                self.final_stage_channel, out_channels, 1, 1, 1,
                norm_layer, act_layer)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
            ('dropout', nn.Dropout(p=dropout, inplace=True)),
            ('classifier', nn.Linear(out_channels, nclass)),
        ]))

    def make_stages(self, layer_infos, block):
        layers = []
        for layer_info in layer_infos:
            layers.extend(self.make_layers(copy.copy(layer_info), block))
        return layers

    def make_layers(self, layer_info, block):
        layers = []
        for _ in range(layer_info.num_layers):
            layers.append(block(layer_info, sd_prob=self.get_sd_prob()))
            layer_info.in_ch = layer_info.out_ch
            layer_info.stride = 1
        return layers

    def get_sd_prob(self):
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_block)
        self.cur_block += 1
        return sd_prob

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def efficientnet_v2_l(nclass=100, in_channels=4):
    structure = [
        (1, 3, 1, 32, 32, 4, False, True),
        (4, 3, 2, 32, 64, 7, False, True),
        (4, 3, 2, 64, 96, 7, False, True),
        (4, 3, 2, 96, 192, 10, True, False),
        (6, 3, 1, 192, 224, 19, True, False),
        (6, 3, 2, 224, 384, 25, True, False),
        (6, 3, 1, 384, 640, 7, True, False),
    ]
    layer_infos = [MBConvConfig(*cfg) for cfg in structure]
    return EfficientNetV2(layer_infos, nclass=nclass, in_channels=in_channels)
