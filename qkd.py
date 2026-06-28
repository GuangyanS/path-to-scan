import os

import torch
import torch.nn.functional as F

from models.efficientnetv2 import efficientnet_v2_l


def resolve_path(path_or_name, root='./checkpoints/'):
    def with_suffix(path):
        return path if path.endswith('.pth') else path + '.pth'

    path = with_suffix(path_or_name)
    rooted = with_suffix(os.path.join(root, path_or_name))
    if os.path.isabs(path_or_name) or os.path.exists(path):
        return path
    return rooted


def prepare_teacher_inputs(inputs, target_size=224):
    if inputs.size(-1) != target_size:
        inputs = F.interpolate(
            inputs, size=(target_size, target_size),
            mode='bilinear', align_corners=False)
    return (inputs - 0.5) / 0.5


def load_teacher(checkpoint_path, num_classes=100, in_channels=4,
                 arch='efficientnet_v2_l', device='cpu'):
    if arch != 'efficientnet_v2_l':
        raise ValueError('Only efficientnet_v2_l is implemented locally')
    teacher = efficientnet_v2_l(
        nclass=num_classes, in_channels=in_channels)

    checkpoint = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)
    state_dict = checkpoint.get('net') or checkpoint.get('state_dict') or checkpoint
    missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print('teacher missing keys: %d, unexpected keys: %d'
              % (len(missing), len(unexpected)))
    print('loaded teacher from %s (ckpt acc=%s, epoch=%s)'
          % (checkpoint_path, checkpoint.get('acc', '?'),
             checkpoint.get('epoch', '?')))
    return teacher.to(device)


def soft_kl_loss(pred_logits, target_logits, temperature=2.0):
    pred_log = F.log_softmax(pred_logits / temperature, dim=1)
    target = F.softmax(target_logits / temperature, dim=1)
    return F.kl_div(pred_log, target, reduction='batchmean') * (temperature ** 2)


def kd_loss(student_logits, teacher_logits, temperature=4.0):
    return soft_kl_loss(student_logits, teacher_logits, temperature)


@torch.no_grad()
def evaluate_teacher(loader, teacher, device):
    teacher.eval()
    num_correct = 0
    num_samples = 0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)
        scores = teacher(prepare_teacher_inputs(x))
        preds = scores.argmax(1)
        num_correct += (preds == y).sum().item()
        num_samples += y.numel()
    acc = float(num_correct) / num_samples
    print('****    Teacher got %d / %d correct (%.2f)'
          % (num_correct, num_samples, 100 * acc))
    return acc
