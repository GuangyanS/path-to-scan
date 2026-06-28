import torch
from torch import nn
import torch.nn.functional as F
from .BasicModule import BasicModule


class ALL_CNN_C(BasicModule):

    def __init__(self, num_classes=100, in_channels=4):
        super(ALL_CNN_C, self).__init__()

        self.model_name = 'ALL_CNN_C'

        self.conv1 = nn.Conv2d(in_channels, 96, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1, bias=False)
        self.dp1 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(96, 192, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1, bias=False)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1, bias=False)
        self.dp2 = nn.Dropout(p=0.5)

        self.conv7 = nn.Conv2d(192, 192, 3, padding=1, bias=False)
        self.conv8 = nn.Conv2d(192, 192, 1, bias=False)
        self.conv9 = nn.Conv2d(192, num_classes, 1)

        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(96)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(192)
        self.bn5 = nn.BatchNorm2d(192)
        self.bn6 = nn.BatchNorm2d(192)
        self.bn7 = nn.BatchNorm2d(192)
        self.bn8 = nn.BatchNorm2d(192)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dp1(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dp2(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.conv9(x)

        x = self.avg(x)
        return torch.flatten(x, 1)


def fake_quant_weight_4bit(weight):
    qmax = 7.0
    scale = weight.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
    scale = torch.clamp(scale / qmax, min=1e-8)
    quantized = torch.clamp(torch.round(weight / scale), -qmax, qmax) * scale
    return weight + (quantized - weight).detach()


class ActivationFakeQuant4Bit(nn.Module):
    def __init__(self, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('amax', torch.tensor(1.0))
        self.register_buffer('initialized', torch.tensor(False))

    def forward(self, x):
        qmax = 15.0
        if self.training:
            observed = x.detach().amax()
            observed = torch.clamp(observed, min=1e-8)
            with torch.no_grad():
                if bool(self.initialized):
                    self.amax.mul_(1.0 - self.momentum).add_(self.momentum * observed)
                else:
                    self.amax.copy_(observed)
                    self.initialized.fill_(True)
            amax = self.amax
        else:
            if bool(self.initialized):
                amax = torch.clamp(self.amax, min=1e-8)
            else:
                amax = torch.clamp(x.detach().amax(), min=1e-8)

        scale = amax / qmax
        quantized = torch.clamp(torch.round(x / scale), 0.0, qmax) * scale
        return x + (quantized - x).detach()


class QuantConv2d4Bit(nn.Conv2d):
    def forward(self, x):
        return F.conv2d(
            x, fake_quant_weight_4bit(self.weight), self.bias, self.stride,
            self.padding, self.dilation, self.groups)


class ALL_CNN_C_QAT(BasicModule):
    """4W4A fake-quantized ALL-CNN-C for QAT fine-tuning."""

    def __init__(self, num_classes=100, in_channels=4):
        super(ALL_CNN_C_QAT, self).__init__()

        self.model_name = 'ALL_CNN_C_QAT'

        self.act0 = ActivationFakeQuant4Bit()
        self.conv1 = QuantConv2d4Bit(in_channels, 96, 3, padding=1, bias=False)
        self.conv2 = QuantConv2d4Bit(96, 96, 3, padding=1, bias=False)
        self.conv3 = QuantConv2d4Bit(96, 96, 3, stride=2, padding=1, bias=False)
        self.dp1 = nn.Dropout(p=0.5)

        self.conv4 = QuantConv2d4Bit(96, 192, 3, padding=1, bias=False)
        self.conv5 = QuantConv2d4Bit(192, 192, 3, padding=1, bias=False)
        self.conv6 = QuantConv2d4Bit(192, 192, 3, stride=2, padding=1, bias=False)
        self.dp2 = nn.Dropout(p=0.5)

        self.conv7 = QuantConv2d4Bit(192, 192, 3, padding=1, bias=False)
        self.conv8 = QuantConv2d4Bit(192, 192, 1, bias=False)
        self.conv9 = QuantConv2d4Bit(192, num_classes, 1)

        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(96)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(192)
        self.bn5 = nn.BatchNorm2d(192)
        self.bn6 = nn.BatchNorm2d(192)
        self.bn7 = nn.BatchNorm2d(192)
        self.bn8 = nn.BatchNorm2d(192)

        self.act1 = ActivationFakeQuant4Bit()
        self.act2 = ActivationFakeQuant4Bit()
        self.act3 = ActivationFakeQuant4Bit()
        self.act4 = ActivationFakeQuant4Bit()
        self.act5 = ActivationFakeQuant4Bit()
        self.act6 = ActivationFakeQuant4Bit()
        self.act7 = ActivationFakeQuant4Bit()
        self.act8 = ActivationFakeQuant4Bit()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.act0(x)

        x = self.act1(F.relu(self.bn1(self.conv1(x))))
        x = self.act2(F.relu(self.bn2(self.conv2(x))))
        x = self.act3(F.relu(self.bn3(self.conv3(x))))
        x = self.dp1(x)

        x = self.act4(F.relu(self.bn4(self.conv4(x))))
        x = self.act5(F.relu(self.bn5(self.conv5(x))))
        x = self.act6(F.relu(self.bn6(self.conv6(x))))
        x = self.dp2(x)

        x = self.act7(F.relu(self.bn7(self.conv7(x))))
        x = self.act8(F.relu(self.bn8(self.conv8(x))))
        x = self.conv9(x)

        x = self.avg(x)
        return torch.flatten(x, 1)


class RealQuantConv2d4Bit(nn.Module):
    """Conv2d with weights stored as signed int4 codes plus FP scales."""

    def __init__(self, conv, bn=None):
        super().__init__()
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        qmax = 7.0
        weight = conv.weight.detach()
        if conv.bias is None:
            bias = torch.zeros(weight.shape[0], device=weight.device)
        else:
            bias = conv.bias.detach()

        if bn is not None:
            std = torch.sqrt(bn.running_var.detach() + bn.eps)
            gain = bn.weight.detach() / std
            weight = weight * gain.view(-1, 1, 1, 1)
            bias = (bias - bn.running_mean.detach()) * gain + bn.bias.detach()

        scale = torch.clamp(
            weight.abs().amax(dim=(1, 2, 3), keepdim=True) / qmax,
            min=1e-8)
        qweight = torch.clamp(torch.round(weight / scale), -qmax, qmax)

        self.register_buffer('qweight', qweight.to(torch.int8))
        self.register_buffer('scale', scale)
        if bn is None and conv.bias is None:
            self.bias = None
        else:
            self.register_buffer('bias', bias.clone())

    def forward(self, x):
        weight = self.qweight.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.conv2d(
            x, weight, bias, self.stride, self.padding, self.dilation,
            self.groups)


class RealQuantActivation4Bit(nn.Module):
    """Activation quantizer that materializes unsigned int4 codes in eval."""

    def __init__(self, fake_quant_module):
        super().__init__()
        amax = fake_quant_module.amax.detach().clone()
        self.register_buffer('amax', torch.clamp(amax, min=1e-8))

    def forward(self, x):
        qmax = 15.0
        scale = self.amax.to(dtype=x.dtype) / qmax
        qact = torch.clamp(torch.round(x / scale), 0.0, qmax).to(torch.uint8)
        return qact.to(dtype=x.dtype) * scale


class ALL_CNN_C_INT4(BasicModule):
    """Eval-only 4W4A model with int4-coded weights and activations."""

    def __init__(self, qat_model):
        super(ALL_CNN_C_INT4, self).__init__()
        self.model_name = 'ALL_CNN_C_INT4'

        self.act0 = RealQuantActivation4Bit(qat_model.act0)
        self.conv1 = RealQuantConv2d4Bit(qat_model.conv1, qat_model.bn1)
        self.conv2 = RealQuantConv2d4Bit(qat_model.conv2, qat_model.bn2)
        self.conv3 = RealQuantConv2d4Bit(qat_model.conv3, qat_model.bn3)
        self.dp1 = nn.Identity()

        self.conv4 = RealQuantConv2d4Bit(qat_model.conv4, qat_model.bn4)
        self.conv5 = RealQuantConv2d4Bit(qat_model.conv5, qat_model.bn5)
        self.conv6 = RealQuantConv2d4Bit(qat_model.conv6, qat_model.bn6)
        self.dp2 = nn.Identity()

        self.conv7 = RealQuantConv2d4Bit(qat_model.conv7, qat_model.bn7)
        self.conv8 = RealQuantConv2d4Bit(qat_model.conv8, qat_model.bn8)
        self.conv9 = RealQuantConv2d4Bit(qat_model.conv9)

        self.act1 = RealQuantActivation4Bit(qat_model.act1)
        self.act2 = RealQuantActivation4Bit(qat_model.act2)
        self.act3 = RealQuantActivation4Bit(qat_model.act3)
        self.act4 = RealQuantActivation4Bit(qat_model.act4)
        self.act5 = RealQuantActivation4Bit(qat_model.act5)
        self.act6 = RealQuantActivation4Bit(qat_model.act6)
        self.act7 = RealQuantActivation4Bit(qat_model.act7)
        self.act8 = RealQuantActivation4Bit(qat_model.act8)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.eval()

    def forward(self, x):
        x = self.act0(x)

        x = self.act1(F.relu(self.conv1(x)))
        x = self.act2(F.relu(self.conv2(x)))
        x = self.act3(F.relu(self.conv3(x)))
        x = self.dp1(x)

        x = self.act4(F.relu(self.conv4(x)))
        x = self.act5(F.relu(self.conv5(x)))
        x = self.act6(F.relu(self.conv6(x)))
        x = self.dp2(x)

        x = self.act7(F.relu(self.conv7(x)))
        x = self.act8(F.relu(self.conv8(x)))
        x = self.conv9(x)

        x = self.avg(x)
        return torch.flatten(x, 1)
