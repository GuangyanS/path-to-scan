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


def _make_feedback_conv(conv):
    output_padding = tuple(max(0, stride - 1) for stride in conv.stride)
    return nn.ConvTranspose2d(
        conv.out_channels, conv.in_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        output_padding=output_padding, bias=False)


def _make_bypass_conv(conv):
    return nn.Conv2d(
        conv.in_channels, conv.out_channels, 1, stride=conv.stride,
        bias=False)


def _alpha_parameter(channels, init_value):
    if init_value < 0:
        raise ValueError('pcn_alpha_init must be >= 0')
    return nn.Parameter(torch.full((channels,), float(init_value)))


def _match_spatial(x, target):
    target_h, target_w = target.shape[-2:]
    if x.size(-2) > target_h:
        x = x[..., :target_h, :]
    if x.size(-1) > target_w:
        x = x[..., :, :target_w]
    pad_h = target_h - x.size(-2)
    pad_w = target_w - x.size(-1)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x


class ALL_CNN_C_PCN(ALL_CNN_C):
    """ALL-CNN-C with local predictive-coding recurrences on conv1-conv8."""

    def __init__(self, num_classes=100, in_channels=4,
                 pcn_cycles=3, pcn_alpha_init=0.000001):
        super(ALL_CNN_C_PCN, self).__init__(num_classes, in_channels)
        self.model_name = 'ALL_CNN_C_PCN'
        self.pcn_cycles = int(pcn_cycles)

        self.fb1 = _make_feedback_conv(self.conv1)
        self.fb2 = _make_feedback_conv(self.conv2)
        self.fb3 = _make_feedback_conv(self.conv3)
        self.fb4 = _make_feedback_conv(self.conv4)
        self.fb5 = _make_feedback_conv(self.conv5)
        self.fb6 = _make_feedback_conv(self.conv6)
        self.fb7 = _make_feedback_conv(self.conv7)
        self.fb8 = _make_feedback_conv(self.conv8)

        self.bp1 = _make_bypass_conv(self.conv1)
        self.bp2 = _make_bypass_conv(self.conv2)
        self.bp3 = _make_bypass_conv(self.conv3)
        self.bp4 = _make_bypass_conv(self.conv4)
        self.bp5 = _make_bypass_conv(self.conv5)
        self.bp6 = _make_bypass_conv(self.conv6)
        self.bp7 = _make_bypass_conv(self.conv7)
        self.bp8 = _make_bypass_conv(self.conv8)

        self.alpha1 = _alpha_parameter(96, pcn_alpha_init)
        self.alpha2 = _alpha_parameter(96, pcn_alpha_init)
        self.alpha3 = _alpha_parameter(96, pcn_alpha_init)
        self.alpha4 = _alpha_parameter(192, pcn_alpha_init)
        self.alpha5 = _alpha_parameter(192, pcn_alpha_init)
        self.alpha6 = _alpha_parameter(192, pcn_alpha_init)
        self.alpha7 = _alpha_parameter(192, pcn_alpha_init)
        self.alpha8 = _alpha_parameter(192, pcn_alpha_init)

    def _pcn_layers(self):
        return (
            (self.conv1, self.bn1, self.fb1, self.bp1, self.alpha1),
            (self.conv2, self.bn2, self.fb2, self.bp2, self.alpha2),
            (self.conv3, self.bn3, self.fb3, self.bp3, self.alpha3),
            (self.conv4, self.bn4, self.fb4, self.bp4, self.alpha4),
            (self.conv5, self.bn5, self.fb5, self.bp5, self.alpha5),
            (self.conv6, self.bn6, self.fb6, self.bp6, self.alpha6),
            (self.conv7, self.bn7, self.fb7, self.bp7, self.alpha7),
            (self.conv8, self.bn8, self.fb8, self.bp8, self.alpha8),
        )

    def init_pcn_extras_from_feedforward(self, zero_bypass=True):
        with torch.no_grad():
            for conv, _, fb, bp, _ in self._pcn_layers():
                fb.weight.copy_(conv.weight)
                if zero_bypass:
                    bp.weight.zero_()

    def _pcn_layer(self, x, conv, bn, fb, bp, alpha):
        state = F.relu(bn(conv(x)))
        alpha = torch.clamp(alpha, min=0.0).view(1, -1, 1, 1)
        for _ in range(self.pcn_cycles):
            pred = _match_spatial(fb(state), x)
            error = F.relu(x - pred)
            state = state + alpha * conv(error)
        bypass = _match_spatial(bp(x), state)
        return state + bypass

    def forward(self, x):
        x = self._pcn_layer(x, self.conv1, self.bn1, self.fb1, self.bp1,
                            self.alpha1)
        x = self._pcn_layer(x, self.conv2, self.bn2, self.fb2, self.bp2,
                            self.alpha2)
        x = self._pcn_layer(x, self.conv3, self.bn3, self.fb3, self.bp3,
                            self.alpha3)
        x = self.dp1(x)

        x = self._pcn_layer(x, self.conv4, self.bn4, self.fb4, self.bp4,
                            self.alpha4)
        x = self._pcn_layer(x, self.conv5, self.bn5, self.fb5, self.bp5,
                            self.alpha5)
        x = self._pcn_layer(x, self.conv6, self.bn6, self.fb6, self.bp6,
                            self.alpha6)
        x = self.dp2(x)

        x = self._pcn_layer(x, self.conv7, self.bn7, self.fb7, self.bp7,
                            self.alpha7)
        x = self._pcn_layer(x, self.conv8, self.bn8, self.fb8, self.bp8,
                            self.alpha8)
        x = self.conv9(x)

        x = self.avg(x)
        return torch.flatten(x, 1)


def fake_quant_weight_4bit(weight):
    qmax = 7.0
    scale = weight.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
    scale = torch.clamp(scale / qmax, min=1e-8)
    quantized = torch.clamp(torch.round(weight / scale), -qmax, qmax) * scale
    return weight + (quantized - weight).detach()


class LUTQCodebook(nn.Module):
    """Shared integer LUT for non-uniform 4-bit weight quantization."""

    def __init__(self, bits=4, int_max=None):
        super().__init__()
        levels = 2 ** (bits - 1) - 1
        max_val = levels if int_max is None else int(int_max)
        if max_val < levels:
            raise ValueError('int_max must be at least %d' % levels)
        self.register_buffer('max_val', torch.tensor(float(max_val)))
        init = torch.linspace(-max_val, max_val, steps=2 * levels + 1)
        self.register_buffer(
            'int_codebook', torch.round(init).to(torch.int32))

    @property
    def codebook_normalized(self):
        return self.int_codebook.to(torch.float32) / torch.clamp(
            self.max_val, min=1e-8)

    @torch.no_grad()
    def update_kmeans(self, weights_norm):
        codebook_norm = self.codebook_normalized
        dists = (weights_norm.unsqueeze(-1) - codebook_norm.unsqueeze(0)).abs()
        idx = dists.argmin(dim=-1)
        new_centroids = torch.zeros(
            self.int_codebook.numel(), device=weights_norm.device,
            dtype=weights_norm.dtype)
        counts = torch.zeros_like(new_centroids)
        new_centroids.scatter_add_(0, idx, weights_norm)
        counts.scatter_add_(0, idx, torch.ones_like(weights_norm))
        active = counts > 0
        new_centroids[active] /= counts[active]
        new_int = torch.round(new_centroids * self.max_val).clamp(
            -self.max_val.item(), self.max_val.item()).to(torch.int32)
        self.int_codebook[active] = new_int[active]
        self.int_codebook.copy_(self.int_codebook.sort().values)


def _weight_group_size(weight, group_size):
    if group_size is None or group_size <= 0:
        return weight.shape[0]
    return max(1, int(group_size))


def _collect_weight_norm(weight, group_size):
    group_size = _weight_group_size(weight, group_size)
    weight = torch.tanh(weight.detach())
    chunks = []
    for start in range(0, weight.shape[0], group_size):
        group = weight[start:start + group_size]
        scale = torch.clamp(group.abs().amax(), min=1e-8)
        chunks.append(torch.clamp(group / scale, -1.0, 1.0).reshape(-1))
    return torch.cat(chunks)


def collect_lutq_normalized_weights(model):
    chunks = []
    if not getattr(model, 'use_lutq', False):
        return torch.zeros(1)
    for conv in model.quant_convs():
        chunks.append(_collect_weight_norm(conv.weight, model.lutq_group_size))
    return torch.cat(chunks) if chunks else torch.zeros(1)


def lutq_fake_quant_weight_4bit(weight, codebook, group_size):
    group_size = _weight_group_size(weight, group_size)
    weight_tanh = torch.tanh(weight)
    chunks = []
    for start in range(0, weight_tanh.shape[0], group_size):
        group = weight_tanh[start:start + group_size]
        scale = torch.clamp(group.detach().abs().amax(), min=1e-8)
        normalized = torch.clamp(group / scale, -1.0, 1.0)
        values = codebook.codebook_normalized.to(
            device=weight.device, dtype=weight.dtype)
        dists = (normalized.reshape(-1).unsqueeze(-1) - values.unsqueeze(0)).abs()
        idx = dists.argmin(dim=-1)
        quantized = values[idx].reshape_as(group) * scale
        chunks.append(quantized)
    quantized = torch.cat(chunks, dim=0)
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


class _PACTClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return torch.clamp(x, min=0.0, max=float(alpha.item()))

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x < 0] = 0
        grad_x[x > alpha] = 0
        grad_alpha = grad_output[x >= alpha].sum().reshape_as(alpha)
        return grad_x, grad_alpha


class PACTActivationFakeQuant4Bit(nn.Module):
    def __init__(self, init_alpha=6.0):
        super().__init__()
        target = torch.tensor(max(float(init_alpha) - 1e-3, 1e-6))
        raw_alpha = torch.log(torch.expm1(target))
        self.alpha = nn.Parameter(raw_alpha)

    @property
    def amax(self):
        return F.softplus(self.alpha) + 1e-3

    def forward(self, x):
        qmax = 15.0
        alpha = self.amax
        x_clip = _PACTClamp.apply(x, alpha)
        scale = alpha / qmax
        quantized = torch.clamp(torch.round(x_clip / scale), 0.0, qmax) * scale
        return x_clip + (quantized - x_clip).detach()


class QuantConv2d4Bit(nn.Conv2d):
    def forward(self, x):
        return F.conv2d(
            x, fake_quant_weight_4bit(self.weight), self.bias, self.stride,
            self.padding, self.dilation, self.groups)


class ALL_CNN_C_QAT(BasicModule):
    """4W4A fake-quantized ALL-CNN-C for QAT fine-tuning."""

    def __init__(self, num_classes=100, in_channels=4,
                 use_lutq=False, lutq_group_size=8, lutq_int_max=7,
                 use_pact=False, pact_alpha=6.0, input_bits=16):
        super(ALL_CNN_C_QAT, self).__init__()

        self.model_name = 'ALL_CNN_C_QAT'
        self.use_lutq = bool(use_lutq)
        self.lutq_group_size = int(lutq_group_size)
        self.lutq_int_max = int(lutq_int_max)
        self.input_bits = int(input_bits)
        self.use_pact = bool(use_pact)
        self.pact_alpha = float(pact_alpha)
        if self.use_lutq:
            self.lutq_codebook = LUTQCodebook(bits=4, int_max=self.lutq_int_max)

        act = (lambda alpha: PACTActivationFakeQuant4Bit(alpha)
               if self.use_pact else ActivationFakeQuant4Bit())

        self.act0 = act(1.0)
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

        self.act1 = act(self.pact_alpha)
        self.act2 = act(self.pact_alpha)
        self.act3 = act(self.pact_alpha)
        self.act4 = act(self.pact_alpha)
        self.act5 = act(self.pact_alpha)
        self.act6 = act(self.pact_alpha)
        self.act7 = act(self.pact_alpha)
        self.act8 = act(self.pact_alpha)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def quant_convs(self):
        return (
            self.conv1, self.conv2, self.conv3, self.conv4, self.conv5,
            self.conv6, self.conv7, self.conv8, self.conv9,
        )

    def _conv_weight(self, conv):
        if self.use_lutq:
            return lutq_fake_quant_weight_4bit(
                conv.weight, self.lutq_codebook, self.lutq_group_size)
        return fake_quant_weight_4bit(conv.weight)

    def _conv(self, conv, x):
        return F.conv2d(
            x, self._conv_weight(conv), conv.bias, conv.stride, conv.padding,
            conv.dilation, conv.groups)

    def _input_quant(self, x):
        if self.input_bits > 4:
            qmax = float(2 ** self.input_bits - 1)
            quantized = torch.clamp(torch.round(x * qmax), 0.0, qmax) / qmax
            return x + (quantized - x).detach()
        return self.act0(x)

    def forward(self, x):
        x = self._input_quant(x)

        x = self.act1(F.relu(self.bn1(self._conv(self.conv1, x))))
        x = self.act2(F.relu(self.bn2(self._conv(self.conv2, x))))
        x = self.act3(F.relu(self.bn3(self._conv(self.conv3, x))))
        x = self.dp1(x)

        x = self.act4(F.relu(self.bn4(self._conv(self.conv4, x))))
        x = self.act5(F.relu(self.bn5(self._conv(self.conv5, x))))
        x = self.act6(F.relu(self.bn6(self._conv(self.conv6, x))))
        x = self.dp2(x)

        x = self.act7(F.relu(self.bn7(self._conv(self.conv7, x))))
        x = self.act8(F.relu(self.bn8(self._conv(self.conv8, x))))
        x = self._conv(self.conv9, x)

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
