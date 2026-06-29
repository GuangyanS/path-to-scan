import torch
from torch import nn
import torch.nn.functional as F


def _activation_scale(fake_quant_module, normalized=False):
    if normalized:
        return torch.tensor(1.0 / 15.0)
    return torch.clamp(fake_quant_module.amax.detach().clone(), min=1e-8) / 15.0


def _requant_scale(fake_quant_module):
    if hasattr(fake_quant_module, 'amax'):
        return torch.clamp(fake_quant_module.amax.detach().clone(), min=1e-8) / 15.0
    return torch.tensor(1.0 / 15.0)


def _pim_chunk_spec(num_rows, keep_rows):
    keep_rows = int(keep_rows)
    if num_rows == 144:
        chunks = [18, 18, 36, 72]
        if keep_rows == 72:
            return chunks, [9, 9, 18, 36]
        if keep_rows == 48:
            return chunks, [6, 6, 12, 24]
        quotas = [max(0, int(round(keep_rows * c / 144.0))) for c in chunks]
        quotas[-1] += keep_rows - sum(quotas)
        return chunks, quotas
    return [num_rows], [max(1, min(num_rows, int(round(num_rows / 3.0))))]


def _row_mask(codes, keep_rows):
    masks = []
    start = 0
    for chunk, quota in zip(*_pim_chunk_spec(codes.shape[1], keep_rows)):
        quota = max(0, min(int(quota), int(chunk)))
        part = codes[:, start:start + chunk]
        if quota == 0:
            masks.append(torch.zeros_like(part, dtype=torch.bool))
            start += chunk
            continue
        n = part.shape[1]
        rank = torch.arange(
            n - 1, -1, -1, device=part.device, dtype=part.dtype).view(1, n, 1)
        tier = (part >= 4).to(part.dtype)
        nz = (part >= 1).to(part.dtype)
        score = tier * (2 * n + 2) + nz * (n + 1) + rank
        top = score.topk(quota, dim=1).indices
        mask = torch.zeros_like(part, dtype=torch.bool)
        mask.scatter_(1, top, True)
        masks.append(mask)
        start += chunk
    return torch.cat(masks, dim=1)


def _midrise_adc(x, keep_rows, adc_bits, noise_sigma=0.0):
    n_codes = float(2 ** int(adc_bits))
    window = float(keep_rows)
    delta = 2.0 * window / n_codes
    code = torch.floor((torch.clamp(x, -window, window) + window) / delta)
    code = torch.clamp(code, 0.0, n_codes - 1.0)
    quantized = -window + (code + 0.5) * delta
    if noise_sigma > 0:
        quantized = quantized + float(noise_sigma) * delta * torch.randn_like(quantized)
    return quantized


def _conv_output_hw(x, conv):
    k_h, k_w = conv.kernel_size
    s_h, s_w = conv.stride
    p_h, p_w = conv.padding
    h = (x.shape[-2] + 2 * p_h - k_h) // s_h + 1
    w = (x.shape[-1] + 2 * p_w - k_w) // s_w + 1
    return h, w


def _bn_scale_bias(conv, bn=None):
    weight = conv.weight.detach()
    if conv.bias is None:
        conv_bias = torch.zeros(weight.shape[0], device=weight.device)
    else:
        conv_bias = conv.bias.detach()
    if bn is None:
        return torch.ones_like(conv_bias), conv_bias
    std = torch.sqrt(bn.running_var.detach() + bn.eps)
    gain = bn.weight.detach() / std
    bias = (conv_bias - bn.running_mean.detach()) * gain + bn.bias.detach()
    return gain, bias


def _lutq_int_weight_scale(qat_model, conv):
    weight = torch.tanh(conv.weight.detach())
    codebook_norm = qat_model.lutq_codebook.codebook_normalized.detach().to(
        device=weight.device, dtype=weight.dtype)
    int_codebook = qat_model.lutq_codebook.int_codebook.detach().round().to(
        device=weight.device, dtype=torch.int32)
    group_size = int(getattr(qat_model, 'lutq_group_size', 8))
    if group_size <= 0:
        group_size = weight.shape[0]

    qint_chunks = []
    scales = torch.empty(
        weight.shape[0], device=weight.device, dtype=torch.float32)
    for start in range(0, weight.shape[0], group_size):
        end = min(start + group_size, weight.shape[0])
        group = weight[start:end]
        scale = torch.clamp(group.abs().amax(), min=1e-8)
        normalized = torch.clamp(group / scale, -1.0, 1.0)
        dists = (normalized.reshape(-1).unsqueeze(-1)
                 - codebook_norm.unsqueeze(0)).abs()
        qindex = dists.argmin(dim=-1).reshape_as(group)
        qint_chunks.append(int_codebook[qindex])
        scales[start:end] = scale
    max_val = qat_model.lutq_codebook.max_val.detach().to(
        device=weight.device, dtype=torch.float32)
    return torch.cat(qint_chunks, dim=0), scales, max_val


def _uniform_int_weight_scale(conv):
    qmax = torch.tensor(7.0, device=conv.weight.device)
    weight = conv.weight.detach()
    scale = torch.clamp(
        weight.abs().amax(dim=(1, 2, 3), keepdim=False) / qmax, min=1e-8)
    qint = torch.clamp(
        torch.round(weight / scale.view(-1, 1, 1, 1)), -7.0, 7.0).to(torch.int32)
    return qint, scale * qmax, qmax


def _int_weight_scale(qat_model, conv):
    if getattr(qat_model, 'use_lutq', False):
        return _lutq_int_weight_scale(qat_model, conv)
    return _uniform_int_weight_scale(conv)


class PIMConvRequant(nn.Module):
    def __init__(self, qat_model, conv, bn, input_scale, requant_scale,
                 masked=True, keep_rows=48, adc_bits=7, noise_sigma=0.0):
        super().__init__()
        qweight, weight_scale, max_val = _int_weight_scale(qat_model, conv)
        bn_scale, bn_bias = _bn_scale_bias(conv, bn)
        self.stride = conv.stride
        self.padding = conv.padding
        self.kernel_size = conv.kernel_size
        self.in_channels = conv.in_channels
        self.masked = bool(masked)
        self.keep_rows = int(keep_rows)
        self.adc_bits = int(adc_bits)
        self.noise_sigma = float(noise_sigma)
        self.unit_channels = max(1, 144 // (conv.kernel_size[0] * conv.kernel_size[1]))
        self.register_buffer('qweight', qweight.to(torch.int32).cpu())
        self.register_buffer('max_val', max_val.to(torch.float32).cpu())
        self.register_buffer('input_scale', input_scale.detach().to(torch.float32).cpu())
        self.register_buffer('requant_scale', requant_scale.detach().to(torch.float32).cpu())
        self.register_buffer(
            'post_scale', (weight_scale * bn_scale).to(torch.float32).cpu())
        self.register_buffer('post_bias', bn_bias.to(torch.float32).cpu())

    def _dense_conv(self, x_code):
        x = x_code.to(torch.float32) * self.input_scale
        weight = self.qweight.to(torch.float32) / self.max_val
        y = F.conv2d(x, weight, None, self.stride, self.padding)
        return y * self.post_scale.view(1, -1, 1, 1) + self.post_bias.view(1, -1, 1, 1)

    def _masked_conv(self, x_code):
        k_h, k_w = self.kernel_size
        patches = F.unfold(
            x_code.to(torch.float32), (k_h, k_w), padding=self.padding,
            stride=self.stride)
        batch = patches.shape[0]
        out_h, out_w = _conv_output_hw(x_code, self)
        output = None
        for start in range(0, self.in_channels, self.unit_channels):
            end = min(start + self.unit_channels, self.in_channels)
            row_start = start * k_h * k_w
            row_end = end * k_h * k_w
            codes = patches[:, row_start:row_end]
            keep = self.keep_rows if codes.shape[1] == 144 else max(
                1, int(round(codes.shape[1] / 3.0)))
            mask = _row_mask(codes, keep).to(torch.float32)
            values = codes * self.input_scale * mask
            weight = (self.qweight[:, start:end].to(torch.float32)
                      / self.max_val).reshape(self.qweight.shape[0], -1)
            partial = torch.einsum('on,bnl->bol', weight, values)
            partial = _midrise_adc(
                partial, keep, self.adc_bits, self.noise_sigma)
            output = partial if output is None else output + partial
        output = output.reshape(batch, self.qweight.shape[0], out_h, out_w)
        return output * self.post_scale.view(1, -1, 1, 1) + self.post_bias.view(1, -1, 1, 1)

    def forward(self, x_code):
        if x_code.device.type != 'cpu':
            raise RuntimeError('PIM simulator runs on CPU; move inputs/model to CPU.')
        y = self._masked_conv(x_code) if self.masked else self._dense_conv(x_code)
        q = torch.round(torch.clamp(y / self.requant_scale, 0.0, 15.0))
        return q.to(torch.int32)


class PIMConvLogits(nn.Module):
    def __init__(self, qat_model, conv, input_scale):
        super().__init__()
        qweight, weight_scale, max_val = _int_weight_scale(qat_model, conv)
        _, bias = _bn_scale_bias(conv, None)
        self.stride = conv.stride
        self.padding = conv.padding
        self.register_buffer('qweight', qweight.to(torch.int32).cpu())
        self.register_buffer('max_val', max_val.to(torch.float32).cpu())
        self.register_buffer('input_scale', input_scale.detach().to(torch.float32).cpu())
        self.register_buffer('post_scale', weight_scale.to(torch.float32).cpu())
        self.register_buffer('bias', bias.to(torch.float32).cpu())

    def forward(self, x_code):
        if x_code.device.type != 'cpu':
            raise RuntimeError('PIM simulator runs on CPU; move inputs/model to CPU.')
        x = x_code.to(torch.float32) * self.input_scale
        weight = self.qweight.to(torch.float32) / self.max_val
        y = F.conv2d(x, weight, None, self.stride, self.padding)
        y = y * self.post_scale.view(1, -1, 1, 1)
        return y + self.bias.view(1, -1, 1, 1)


class ALLCNNPIMSimulator(nn.Module):
    """ScAN-PCN PIM conv simulator for ALL_CNN_C_QAT-compatible checkpoints."""

    def __init__(self, qat_model, adc_bits=7, keep_rows=48,
                 stage11_keep_rows=72, noise_sigma=0.0):
        super().__init__()
        qat_model.eval()
        self.input_bits = int(getattr(qat_model, 'input_bits', 16))
        self.input_qmax = float(2 ** self.input_bits - 1)
        normalized = bool(getattr(qat_model, 'pim_normalized_activations', False))
        s0 = torch.tensor(1.0 / self.input_qmax)
        s1 = _activation_scale(qat_model.act1, normalized)
        s2 = _activation_scale(qat_model.act2, normalized)
        s3 = _activation_scale(qat_model.act3, normalized)
        s4 = _activation_scale(qat_model.act4, normalized)
        s5 = _activation_scale(qat_model.act5, normalized)
        s6 = _activation_scale(qat_model.act6, normalized)
        s7 = _activation_scale(qat_model.act7, normalized)
        s8 = _activation_scale(qat_model.act8, normalized)
        q1 = _requant_scale(qat_model.act1)
        q2 = _requant_scale(qat_model.act2)
        q3 = _requant_scale(qat_model.act3)
        q4 = _requant_scale(qat_model.act4)
        q5 = _requant_scale(qat_model.act5)
        q6 = _requant_scale(qat_model.act6)
        q7 = _requant_scale(qat_model.act7)
        q8 = _requant_scale(qat_model.act8)
        self.normalized_activations = normalized
        self.adc_bits = int(adc_bits)
        self.keep_rows = int(keep_rows)
        self.stage11_keep_rows = int(stage11_keep_rows)
        self.noise_sigma = float(noise_sigma)

        self.conv1 = PIMConvRequant(
            qat_model, qat_model.conv1, qat_model.bn1, s0, q1,
            masked=False, keep_rows=keep_rows, adc_bits=adc_bits,
            noise_sigma=noise_sigma)
        self.conv2 = PIMConvRequant(
            qat_model, qat_model.conv2, qat_model.bn2, s1, q2,
            masked=True, keep_rows=stage11_keep_rows, adc_bits=adc_bits,
            noise_sigma=noise_sigma)
        self.conv3 = PIMConvRequant(
            qat_model, qat_model.conv3, qat_model.bn3, s2, q3,
            masked=True, keep_rows=keep_rows, adc_bits=adc_bits,
            noise_sigma=noise_sigma)
        self.conv4 = PIMConvRequant(
            qat_model, qat_model.conv4, qat_model.bn4, s3, q4,
            masked=True, keep_rows=keep_rows, adc_bits=adc_bits,
            noise_sigma=noise_sigma)
        self.conv5 = PIMConvRequant(
            qat_model, qat_model.conv5, qat_model.bn5, s4, q5,
            masked=True, keep_rows=keep_rows, adc_bits=adc_bits,
            noise_sigma=noise_sigma)
        self.conv6 = PIMConvRequant(
            qat_model, qat_model.conv6, qat_model.bn6, s5, q6,
            masked=True, keep_rows=keep_rows, adc_bits=adc_bits,
            noise_sigma=noise_sigma)
        self.conv7 = PIMConvRequant(
            qat_model, qat_model.conv7, qat_model.bn7, s6, q7,
            masked=True, keep_rows=keep_rows, adc_bits=adc_bits,
            noise_sigma=noise_sigma)
        self.conv8 = PIMConvRequant(
            qat_model, qat_model.conv8, qat_model.bn8, s7, q8,
            masked=True, keep_rows=keep_rows, adc_bits=adc_bits,
            noise_sigma=noise_sigma)
        self.conv9 = PIMConvLogits(qat_model, qat_model.conv9, s8)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        if getattr(qat_model, 'use_lutq', False):
            self.register_buffer(
                'lutq_int_codebook',
                qat_model.lutq_codebook.int_codebook.detach().round().to(torch.int32).cpu())

    def quantize_input(self, x):
        x = x.to(torch.float32).cpu()
        q = torch.round(x * self.input_qmax)
        return torch.clamp(q, 0.0, self.input_qmax).to(torch.int32)

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
        for module in self.modules():
            if isinstance(module, (PIMConvRequant, PIMConvLogits)):
                wmins.append(int(module.qweight.min().item()))
                wmaxs.append(int(module.qweight.max().item()))
        result = {
            'input': (0, int(self.input_qmax)),
            'weight': (min(wmins), max(wmaxs)),
            'activation': (0, 15),
            'adc_bits': self.adc_bits,
            'masked_keep_rows': self.keep_rows,
            'stage1.1_keep_rows': self.stage11_keep_rows,
            'normalized_activations': self.normalized_activations,
        }
        if hasattr(self, 'lutq_int_codebook'):
            result['weight_index'] = (0, int(self.lutq_int_codebook.numel() - 1))
            result['weight_lut_int'] = [
                int(x) for x in self.lutq_int_codebook.tolist()]
        return result
