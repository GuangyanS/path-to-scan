import torch
from torch import nn
import torch.nn.functional as F


def _activation_scale(fake_quant_module):
    return torch.clamp(fake_quant_module.amax.detach().clone(), min=1e-8) / 15.0


def _fold_conv_bn(conv, bn=None):
    weight = conv.weight.detach()
    if conv.bias is None:
        bias = torch.zeros(weight.shape[0], device=weight.device)
    else:
        bias = conv.bias.detach()

    if bn is None:
        return weight, bias

    std = torch.sqrt(bn.running_var.detach() + bn.eps)
    gain = bn.weight.detach() / std
    weight = weight * gain.view(-1, 1, 1, 1)
    bias = (bias - bn.running_mean.detach()) * gain + bn.bias.detach()
    return weight, bias


def _qat_weight(qat_model, conv):
    if getattr(qat_model, 'use_lutq', False):
        return qat_model._conv_weight(conv).detach()
    qmax = 7.0
    weight = conv.weight.detach()
    scale = torch.clamp(
        weight.abs().amax(dim=(1, 2, 3), keepdim=True) / qmax,
        min=1e-8)
    return torch.clamp(torch.round(weight / scale), -qmax, qmax) * scale


def _fold_qat_conv_bn(qat_model, conv, bn=None):
    weight = _qat_weight(qat_model, conv)
    if conv.bias is None:
        bias = torch.zeros(weight.shape[0], device=weight.device)
    else:
        bias = conv.bias.detach()

    if bn is None:
        return weight, bias

    std = torch.sqrt(bn.running_var.detach() + bn.eps)
    gain = bn.weight.detach() / std
    weight = weight * gain.view(-1, 1, 1, 1)
    bias = (bias - bn.running_mean.detach()) * gain + bn.bias.detach()
    return weight, bias


def _lutq_indices_scale_bias(qat_model, conv, bn=None):
    # Precompute the integer LUT indices and per-channel scales from the QAT
    # checkpoint. Runtime simulator forward uses only the materialized int32
    # qweight and the precomputed scales.
    weight = torch.tanh(conv.weight.detach())
    codebook_norm = qat_model.lutq_codebook.codebook_normalized.detach().to(
        device=weight.device, dtype=weight.dtype)
    group_size = int(getattr(qat_model, 'lutq_group_size', 8))
    if group_size <= 0:
        group_size = weight.shape[0]

    qindex_chunks = []
    scales = torch.empty(
        weight.shape[0], 1, 1, 1, device=weight.device, dtype=weight.dtype)
    for start in range(0, weight.shape[0], group_size):
        end = min(start + group_size, weight.shape[0])
        group = weight[start:end]
        scale = torch.clamp(group.abs().amax(), min=1e-8)
        normalized = torch.clamp(group / scale, -1.0, 1.0)
        dists = (normalized.reshape(-1).unsqueeze(-1)
                 - codebook_norm.unsqueeze(0)).abs()
        qindex = dists.argmin(dim=-1).reshape_as(group)
        qindex_chunks.append(qindex)
        scales[start:end] = scale

    qindex = torch.cat(qindex_chunks, dim=0).to(torch.int64)
    if conv.bias is None:
        bias = torch.zeros(weight.shape[0], device=weight.device)
    else:
        bias = conv.bias.detach()

    if bn is None:
        return qindex, scales, bias

    std = torch.sqrt(bn.running_var.detach() + bn.eps)
    gain = bn.weight.detach() / std
    scales = scales * gain.view(-1, 1, 1, 1)
    bias = (bias - bn.running_mean.detach()) * gain + bn.bias.detach()
    return qindex, scales, bias


class Int4ConvRequant(nn.Module):
    """Integer conv followed by ReLU and requantization to uint4 activation."""

    def __init__(self, qat_model, conv, bn, input_scale, output_scale):
        super().__init__()
        weight, bias = _fold_qat_conv_bn(qat_model, conv, bn)
        qmax = 7.0
        weight_scale = torch.clamp(
            weight.abs().amax(dim=(1, 2, 3), keepdim=True) / qmax,
            min=1e-8)
        qweight = torch.clamp(torch.round(weight / weight_scale), -qmax, qmax)

        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.register_buffer('qweight', qweight.to(torch.int32).cpu())
        self.register_buffer('weight_scale', weight_scale.cpu())
        self.register_buffer('requant_scale',
                             (input_scale * weight_scale.view(-1) / output_scale).cpu())
        self.register_buffer('bias_scaled', (bias / output_scale).cpu())

    def forward(self, x_code):
        if x_code.device.type != 'cpu':
            raise RuntimeError('Int4 simulator uses CPU int32 conv2d; move inputs/model to CPU.')
        acc = F.conv2d(
            x_code.to(torch.int32), self.qweight, None, self.stride,
            self.padding, self.dilation, self.groups)
        y = acc.to(torch.float32) * self.requant_scale.view(1, -1, 1, 1)
        y = y + self.bias_scaled.view(1, -1, 1, 1)
        return torch.clamp(torch.round(y), 0.0, 15.0).to(torch.int32)


class Int4ConvLogits(nn.Module):
    """Integer conv with float dequantized logits output."""

    def __init__(self, qat_model, conv, input_scale):
        super().__init__()
        weight, bias = _fold_qat_conv_bn(qat_model, conv, None)
        qmax = 7.0
        weight_scale = torch.clamp(
            weight.abs().amax(dim=(1, 2, 3), keepdim=True) / qmax,
            min=1e-8)
        qweight = torch.clamp(torch.round(weight / weight_scale), -qmax, qmax)

        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.register_buffer('qweight', qweight.to(torch.int32).cpu())
        self.register_buffer('output_scale',
                             (input_scale * weight_scale.view(-1)).cpu())
        self.register_buffer('bias', bias.cpu())

    def forward(self, x_code):
        if x_code.device.type != 'cpu':
            raise RuntimeError('Int4 simulator uses CPU int32 conv2d; move inputs/model to CPU.')
        acc = F.conv2d(
            x_code.to(torch.int32), self.qweight, None, self.stride,
            self.padding, self.dilation, self.groups)
        y = acc.to(torch.float32) * self.output_scale.view(1, -1, 1, 1)
        return y + self.bias.view(1, -1, 1, 1)


class LUTQConvRequant(nn.Module):
    """Integer-LUT weight conv followed by uint4 activation requantization."""

    def __init__(self, qat_model, conv, bn, input_scale, output_scale):
        super().__init__()
        qindex, channel_scale, bias = _lutq_indices_scale_bias(qat_model, conv, bn)
        int_codebook = qat_model.lutq_codebook.int_codebook.detach().round().to(
            torch.int32)
        qweight = int_codebook[qindex]
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.register_buffer('qindex', qindex.to(torch.int64).cpu())
        self.register_buffer('qweight', qweight.to(torch.int32).cpu())
        self.register_buffer(
            'max_val',
            qat_model.lutq_codebook.max_val.detach().to(torch.float32).cpu())
        self.register_buffer(
            'requant_scale',
            (input_scale * channel_scale.view(-1)
             / qat_model.lutq_codebook.max_val.detach()
             / output_scale).cpu())
        self.register_buffer('bias_scaled', (bias / output_scale).cpu())

    def forward(self, x_code):
        if x_code.device.type != 'cpu':
            raise RuntimeError('Int4 simulator uses CPU int32 conv2d; move inputs/model to CPU.')
        acc = F.conv2d(
            x_code.to(torch.int32), self.qweight, None, self.stride, self.padding,
            self.dilation, self.groups)
        y = acc.to(torch.float32) * self.requant_scale.view(1, -1, 1, 1)
        y = y + self.bias_scaled.view(1, -1, 1, 1)
        return torch.clamp(torch.round(y), 0.0, 15.0).to(torch.int32)


class LUTQConvLogits(nn.Module):
    """Integer-LUT weight conv with dequantized FP32 logits output."""

    def __init__(self, qat_model, conv, input_scale):
        super().__init__()
        qindex, channel_scale, bias = _lutq_indices_scale_bias(qat_model, conv, None)
        int_codebook = qat_model.lutq_codebook.int_codebook.detach().round().to(
            torch.int32)
        qweight = int_codebook[qindex]
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.register_buffer('qindex', qindex.to(torch.int64).cpu())
        self.register_buffer('qweight', qweight.to(torch.int32).cpu())
        self.register_buffer(
            'output_scale',
            (input_scale * channel_scale.view(-1)
             / qat_model.lutq_codebook.max_val.detach()).cpu())
        self.register_buffer('bias', bias.cpu())

    def forward(self, x_code):
        if x_code.device.type != 'cpu':
            raise RuntimeError('Int4 simulator uses CPU int32 conv2d; move inputs/model to CPU.')
        acc = F.conv2d(
            x_code.to(torch.int32), self.qweight, None, self.stride, self.padding,
            self.dilation, self.groups)
        y = acc.to(torch.float32) * self.output_scale.view(1, -1, 1, 1)
        return y + self.bias.view(1, -1, 1, 1)


class ALLCNNInt4Simulator(nn.Module):
    """W4A4 integer-domain simulator for the current ALL_CNN_C_QAT model.

    Weights are signed int4 codes in [-7, 7]. Activations are unsigned int4
    codes in [0, 15]. Conv layers use int32 accumulation, then a precomputed
    requantization scale maps the accumulator to the next activation code.
    """

    def __init__(self, qat_model):
        super().__init__()
        qat_model.eval()

        s0 = _activation_scale(qat_model.act0)
        s1 = _activation_scale(qat_model.act1)
        s2 = _activation_scale(qat_model.act2)
        s3 = _activation_scale(qat_model.act3)
        s4 = _activation_scale(qat_model.act4)
        s5 = _activation_scale(qat_model.act5)
        s6 = _activation_scale(qat_model.act6)
        s7 = _activation_scale(qat_model.act7)
        s8 = _activation_scale(qat_model.act8)

        self.register_buffer('input_scale', s0.cpu())
        self.use_lutq = bool(getattr(qat_model, 'use_lutq', False))
        if self.use_lutq:
            self.register_buffer(
                'lutq_int_codebook',
                qat_model.lutq_codebook.int_codebook.detach().round().to(torch.int32).cpu())
            conv_requant = LUTQConvRequant
            conv_logits = LUTQConvLogits
        else:
            conv_requant = Int4ConvRequant
            conv_logits = Int4ConvLogits

        self.conv1 = conv_requant(qat_model, qat_model.conv1, qat_model.bn1, s0, s1)
        self.conv2 = conv_requant(qat_model, qat_model.conv2, qat_model.bn2, s1, s2)
        self.conv3 = conv_requant(qat_model, qat_model.conv3, qat_model.bn3, s2, s3)
        self.conv4 = conv_requant(qat_model, qat_model.conv4, qat_model.bn4, s3, s4)
        self.conv5 = conv_requant(qat_model, qat_model.conv5, qat_model.bn5, s4, s5)
        self.conv6 = conv_requant(qat_model, qat_model.conv6, qat_model.bn6, s5, s6)
        self.conv7 = conv_requant(qat_model, qat_model.conv7, qat_model.bn7, s6, s7)
        self.conv8 = conv_requant(qat_model, qat_model.conv8, qat_model.bn8, s7, s8)
        self.conv9 = conv_logits(qat_model, qat_model.conv9, s8)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def quantize_input(self, x):
        x = x.to(torch.float32).cpu()
        q = torch.round(x / self.input_scale.to(dtype=x.dtype))
        return torch.clamp(q, 0.0, 15.0).to(torch.int32)

    def forward(self, x):
        x = self.quantize_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.avg(x)
        return torch.flatten(x, 1)

    def code_ranges(self):
        wmins, wmaxs = [], []
        if self.use_lutq:
            for module in self.modules():
                if isinstance(module, (LUTQConvRequant, LUTQConvLogits)):
                    wmins.append(int(module.qweight.min().item()))
                    wmaxs.append(int(module.qweight.max().item()))
            return {
                'weight_index': (0, int(self.lutq_int_codebook.numel() - 1)),
                'weight_lut_int': [int(x) for x in self.lutq_int_codebook.tolist()],
                'weight': (min(wmins), max(wmaxs)),
                'activation': (0, 15),
            }
        for module in self.modules():
            if isinstance(module, (Int4ConvRequant, Int4ConvLogits)):
                wmins.append(int(module.qweight.min().item()))
                wmaxs.append(int(module.qweight.max().item()))
        return {
            'weight': (min(wmins), max(wmaxs)),
            'activation': (0, 15),
        }
