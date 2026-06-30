import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import opt
from h5_dataset import CIFAR100PythonDataset, RawH5CIFARDataset
import models
import qkd
from simulator import ALLCNNInt4Simulator, ALLCNNPIMSimulator


def set_seed(seed):
    """Make a run reproducible across python / numpy / torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_acc(loader, model, device, name='test', max_batches=None):
    print('* * *  Checking accuracy on %s set' % name)
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum().item()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('****    Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def build_h5_loaders(return_index=False):
    if opt.dataset == 'cifar100_rgb':
        train_set = CIFAR100PythonDataset(
            opt.cifar100_path, train=True, augment=opt.raw_augment,
            return_index=return_index)
        test_set = CIFAR100PythonDataset(
            opt.cifar100_path, train=False, augment=False)
    else:
        train_set = RawH5CIFARDataset(
            opt.h5_path, train=True, augment=opt.raw_augment,
            noise=opt.raw_noise, return_index=return_index)
        test_set = RawH5CIFARDataset(
            opt.h5_path, train=False, augment=False,
            noise=opt.raw_noise)
    generator = torch.Generator()
    generator.manual_seed(opt.seed)
    persistent_workers = opt.num_workers > 0
    loader_train = DataLoader(
        train_set, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=opt.use_gpu,
        worker_init_fn=seed_worker, generator=generator,
        persistent_workers=persistent_workers)
    loader_test = DataLoader(
        test_set, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=opt.use_gpu,
        worker_init_fn=seed_worker, persistent_workers=persistent_workers)
    return loader_train, loader_test


def get_device():
    if opt.use_gpu == True and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def checkpoint_path(name):
    path = opt.test_model_path + name
    if not path.endswith('.pth'):
        path += '.pth'
    return path


def checkpoint_best_acc(name):
    path = checkpoint_path(name)
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location='cpu')
    testacc = checkpoint.get('testacc', [])
    if not testacc:
        return None
    return max(float(x) for x in testacc)


def build_model():
    kwargs = {
        'num_classes': opt.num_classes,
        'in_channels': opt.input_channels,
    }
    if opt.model == 'ALL_CNN_C_PCN':
        kwargs['pcn_cycles'] = opt.pcn_cycles
        kwargs['pcn_alpha_init'] = opt.pcn_alpha_init
    return getattr(models, opt.model)(**kwargs)


def load_classifier_checkpoint(model, cp_name):
    incompatible = model.load(
        opt.test_model_path + cp_name, strict=opt.checkpoint_load_strict)
    missing = getattr(incompatible, 'missing_keys', ())
    missing_pcn = any(
        k.startswith(('fb', 'bp', 'alpha')) for k in missing)
    if (missing_pcn and opt.pcn_init_extras_from_ff
            and hasattr(model, 'init_pcn_extras_from_feedforward')):
        model.init_pcn_extras_from_feedforward()
        print('initialized PCN feedback/bypass from feedforward weights')
    return incompatible


QAT_STAGE_MAP = (
    ('stage1.0', 'conv1', 'bn1', 'act1'),
    ('stage1.1', 'conv2', 'bn2', 'act2'),
    ('stage1.2', 'conv3', 'bn3', 'act3'),
    ('stage2.0', 'conv4', 'bn4', 'act4'),
    ('stage2.1', 'conv5', 'bn5', 'act5'),
    ('stage2.2', 'conv6', 'bn6', 'act6'),
    ('stage3.0', 'conv7', 'bn7', 'act7'),
    ('stage3.1', 'conv8', 'bn8', 'act8'),
    ('stage3.2', 'conv9', None, None),
)


def _legacy_qat_state_dict(legacy_sd, model):
    state = model.state_dict()
    converted = dict(state)
    for stage, conv, bn, act in QAT_STAGE_MAP:
        converted[conv + '.weight'] = legacy_sd[stage + '.conv.weight']
        bias_key = stage + '.conv.bias'
        if bias_key in legacy_sd and conv + '.bias' in state:
            converted[conv + '.bias'] = legacy_sd[bias_key]

        if bn is not None:
            for suffix in ('weight', 'bias', 'running_mean', 'running_var',
                           'num_batches_tracked'):
                old_key = stage + '.norm.' + suffix
                new_key = bn + '.' + suffix
                if old_key in legacy_sd and new_key in state:
                    converted[new_key] = legacy_sd[old_key]

        pact_key = stage + '.pact.alpha'
        act_key = None if act is None else act + '.alpha'
        if pact_key in legacy_sd and act_key in state:
            converted[act_key] = legacy_sd[pact_key]

    if getattr(model, 'use_lutq', False):
        codebook_key = 'stage1.0.codebook.int_codebook'
        max_key = 'stage1.0.codebook.max_val'
        if codebook_key in legacy_sd and 'lutq_codebook.int_codebook' in state:
            converted['lutq_codebook.int_codebook'] = legacy_sd[codebook_key].to(
                state['lutq_codebook.int_codebook'].dtype)
        if max_key in legacy_sd and 'lutq_codebook.max_val' in state:
            converted['lutq_codebook.max_val'] = legacy_sd[max_key].to(
                state['lutq_codebook.max_val'].dtype)
    return converted


def load_qat_checkpoint(model, cp_name, strict=True):
    path = resolve_checkpoint_path(cp_name)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if hasattr(model, 'input_bits') and 'input_bits' in checkpoint:
        model.input_bits = int(checkpoint['input_bits'])
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'qat_state_dict' in checkpoint:
        state_dict = checkpoint['qat_state_dict']
        if any(k.startswith('stage1.') for k in state_dict):
            state_dict = _legacy_qat_state_dict(state_dict, model)
    else:
        state_dict = checkpoint

    incompatible = model.load_state_dict(state_dict, strict=strict)
    model.epoch = checkpoint.get('epoch', 0)
    model.lr_pre_epoch = checkpoint.get('lr', [])
    model.loss_pre_epoch = checkpoint.get('loss', [])
    if 'testacc' in checkpoint:
        model.testacc_pre_epoch = checkpoint['testacc']
    elif 'acc' in checkpoint:
        acc = float(checkpoint['acc'])
        model.testacc_pre_epoch = [acc / 100.0 if acc > 1.0 else acc]
    else:
        model.testacc_pre_epoch = []
    print('loaded QAT checkpoint from %s' % path)
    if not strict:
        print('loaded with missing keys: %d, unexpected keys: %d'
              % (len(incompatible.missing_keys),
                 len(incompatible.unexpected_keys)))
    return incompatible


def resolve_checkpoint_path(path_or_name):
    return qkd.resolve_path(path_or_name, opt.test_model_path)


def parse_milestones(value):
    if isinstance(value, (tuple, list)):
        return [int(x) for x in value]
    return [int(x) for x in str(value).split(',') if x]


def make_scheduler(optimizer, milestones, warmup):
    if warmup > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup)
        main_sched = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, main_sched],
            milestones=[warmup])
    else:
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)


def fit(model, loader_train, loader_test, device, max_epoch, lr, milestones,
        warmup, checkpoint_save_name):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=opt.weight_decay, nesterov=True)
    scheduler = make_scheduler(optimizer, milestones, warmup)

    best_testacc = 0.0
    for epoch in range(max_epoch):
        model.train()
        for ii, (data, label) in tqdm(enumerate(loader_train)):
            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            scores = model(data)
            loss = F.cross_entropy(scores, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_now = loss.item()

            if ii == 0:
                print('Epoch [{}/{}], Loss: {:.4f}, lr :{: f}'
                      .format(epoch + 1, max_epoch, loss_now,
                              optimizer.param_groups[0]['lr']))

        scheduler.step()

        if (epoch + 1) % opt.eval_every == 0:
            testacc = check_acc(loader_test, model, device, name='test')
            model.update_epoch(optimizer.param_groups[0]['lr'], loss_now, testacc)

            if testacc > best_testacc:
                best_testacc = testacc
                saved = model.save(checkpoint_save_name)
                print('**  New best test acc %.2f%% -> saved %s' % (testacc * 100, saved))


def qkd_stage_defaults(stage):
    stage = stage.upper()
    defaults = {
        'SS': {'epochs': 30, 'lr': 0.01, 'milestones': '10,20',
               'temperature': 4.0},
        'CS': {'epochs': 100, 'lr': 0.01, 'milestones': '80,120',
               'temperature': 2.0},
        'TU': {'epochs': 70, 'lr': 0.005, 'milestones': '40,60',
               'temperature': 4.0},
    }
    if stage not in defaults:
        raise ValueError('qkd_stage must be SS, CS, or TU')
    return defaults[stage]


def qkd_default_student_checkpoint(stage):
    if opt.qkd_student_checkpoint_name:
        return opt.qkd_student_checkpoint_name
    name = opt.qkd_checkpoint_save_name
    if opt.qkd_use_lutq:
        name += '_lutq'
    if stage == 'CS':
        return name + '_ss'
    if stage == 'TU':
        return name + '_cs'
    return opt.qat_checkpoint_save_name


def qkd_stage_save_name(stage):
    name = opt.qkd_checkpoint_save_name
    if opt.qkd_use_lutq:
        name += '_lutq'
    return name + '_' + stage.lower()


def make_qat_model():
    cls = models.ALL_CNN_C_PIM_QAT if opt.pim_qat else models.ALL_CNN_C_QAT
    extra = {}
    if opt.pim_qat:
        extra = {
            'pim_adc_bits': opt.pim_adc_bits,
            'pim_keep': opt.pim_keep,
            'pim_stage11_keep': opt.pim_stage11_keep,
            'pim_noise_sigma': opt.pim_noise_sigma,
        }
    return cls(
        num_classes=opt.num_classes, in_channels=opt.input_channels,
        use_lutq=opt.qkd_use_lutq, lutq_group_size=opt.lutq_group_size,
        lutq_int_max=opt.lutq_int_max, use_pact=opt.use_pact,
        pact_alpha=opt.pact_alpha, input_bits=opt.input_bits, **extra)


def update_lutq_codebook(model):
    if getattr(model, 'use_lutq', False):
        weights = models.collect_lutq_normalized_weights(model)
        model.lutq_codebook.update_kmeans(weights)


@torch.no_grad()
def precompute_teacher_logits(teacher, device):
    cache_set = RawH5CIFARDataset(
        opt.h5_path, train=True, augment=False, noise=opt.raw_noise)
    cache_loader = DataLoader(
        cache_set, batch_size=opt.batch_size, shuffle=False,
        num_workers=0, pin_memory=opt.use_gpu)
    teacher.eval()
    chunks = []
    print('precomputing frozen teacher logits for TU cache')
    for data, _ in tqdm(cache_loader):
        data = data.to(device=device, dtype=torch.float32)
        scores = teacher(qkd.prepare_teacher_inputs(data))
        chunks.append(scores.detach().cpu())
    logits = torch.cat(chunks, dim=0)
    print('cached teacher logits:', tuple(logits.shape))
    return logits


def fit_qkd(stage, student, teacher, loader_train, loader_test, device,
            checkpoint_save_name, initial_testacc=None):
    defaults = qkd_stage_defaults(stage)
    max_epoch = opt.qkd_max_epoch if opt.qkd_max_epoch is not None else defaults['epochs']
    lr = opt.qkd_lr if opt.qkd_lr is not None else defaults['lr']
    milestones = opt.qkd_milestones if opt.qkd_milestones is not None else defaults['milestones']
    temperature = opt.qkd_temperature if opt.qkd_temperature is not None else defaults['temperature']
    milestones = parse_milestones(milestones)

    params = [{'params': student.parameters(), 'lr': lr}]
    if stage == 'CS':
        params.append({
            'params': teacher.parameters(),
            'lr': lr * opt.qkd_teacher_lr_factor,
        })
    optimizer = optim.SGD(
        params, lr=lr, momentum=0.9, weight_decay=opt.weight_decay,
        nesterov=True)
    scheduler = make_scheduler(optimizer, milestones, 2)

    existing_best = checkpoint_best_acc(checkpoint_save_name)
    best_testacc = initial_testacc if initial_testacc is not None else 0.0
    if existing_best is not None:
        best_testacc = max(best_testacc, existing_best)
    if initial_testacc is not None:
        if existing_best is None or initial_testacc >= existing_best:
            student.update_epoch(lr, 0.0, initial_testacc)
            saved = student.save(checkpoint_save_name)
            print('**  Initial QKD %s baseline %.2f%% -> saved %s'
                  % (stage, initial_testacc * 100, saved))
        else:
            print('**  Keep existing QKD %s best %.2f%% over baseline %.2f%%'
                  % (stage, existing_best * 100, initial_testacc * 100))

    teacher_logits_cache = None
    if stage == 'TU' and teacher is not None and opt.qkd_cache_teacher_logits:
        teacher_logits_cache = precompute_teacher_logits(teacher, device)

    for epoch in range(max_epoch):
        student.train()
        if teacher is not None:
            teacher.train(stage == 'CS')

        loss_now = 0.0
        for ii, batch in tqdm(enumerate(loader_train)):
            if len(batch) == 3:
                data, label, index = batch
            else:
                data, label = batch
                index = None
            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            if stage == 'SS':
                update_lutq_codebook(student)
            optimizer.zero_grad()
            student_scores = student(data)
            student_ce = F.cross_entropy(student_scores, label)
            loss = student_ce

            if teacher is not None:
                t_data = qkd.prepare_teacher_inputs(data)
                if stage == 'CS':
                    teacher_scores = teacher(t_data)
                    if opt.qkd_loss == 'dkd':
                        student_kd = qkd.dkd_loss(
                            student_scores, teacher_scores, label,
                            temperature, opt.dkd_alpha, opt.dkd_beta)
                    else:
                        student_kd = qkd.soft_kl_loss(
                            student_scores, teacher_scores, temperature)
                    alpha = opt.qkd_kd_weight
                    loss = (1.0 - alpha) * student_ce + alpha * student_kd
                else:
                    if teacher_logits_cache is not None and index is not None:
                        teacher_scores = teacher_logits_cache[index].to(
                            device=device, dtype=torch.float32)
                    else:
                        with torch.no_grad():
                            teacher_scores = teacher(t_data)
                    if opt.qkd_loss == 'dkd':
                        student_kd = qkd.dkd_loss(
                            student_scores, teacher_scores, label,
                            temperature, opt.dkd_alpha, opt.dkd_beta)
                    else:
                        student_kd = qkd.soft_kl_loss(
                            student_scores, teacher_scores, temperature)
                    alpha = opt.qkd_kd_weight
                    loss = (1.0 - alpha) * student_ce + alpha * student_kd

            loss.backward()
            if opt.qkd_grad_clip and opt.qkd_grad_clip > 0:
                params_for_clip = []
                for group in optimizer.param_groups:
                    params_for_clip.extend(group['params'])
                torch.nn.utils.clip_grad_norm_(params_for_clip, opt.qkd_grad_clip)
            optimizer.step()
            loss_now = loss.item()

            if ii == 0:
                print('QKD {} Epoch [{}/{}], Loss: {:.4f}, lr :{: f}'
                      .format(stage, epoch + 1, max_epoch, loss_now,
                              optimizer.param_groups[0]['lr']))

        scheduler.step()

        if (epoch + 1) % opt.eval_every == 0:
            testacc = check_acc(loader_test, student, device, name='test')
            student.update_epoch(
                optimizer.param_groups[0]['lr'], loss_now, testacc)
            improved = testacc > best_testacc
            if improved:
                best_testacc = testacc
                saved = student.save(checkpoint_save_name)
                print('**  New best QKD %s acc %.2f%% -> saved %s'
                      % (stage, testacc * 100, saved))
            if stage == 'CS' and teacher is not None and (improved or epoch == 0):
                teacher_path = checkpoint_path(opt.qkd_teacher_save_name)
                torch.save({
                    'net': teacher.state_dict(),
                    'acc': testacc * 100,
                    'epoch': epoch + 1,
                }, teacher_path)
                print('**  Saved QKD teacher %s' % teacher_path)


def train(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    loader_train, loader_test = build_h5_loaders()
    device = get_device()

    model = build_model()
    if opt.use_trained_model == True:
        cp_name = opt.checkpoint_load_name or opt.model
        load_classifier_checkpoint(model, cp_name)
    model.to(device)

    fit(model, loader_train, loader_test, device, opt.max_epoch, opt.lr,
        parse_milestones(opt.milestones), opt.warmup, opt.checkpoint_save_name)


def test(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    _, loader_test = build_h5_loaders()

    device = get_device()

    model = build_model()

    cp_name = opt.checkpoint_load_name or opt.checkpoint_save_name or opt.model
    load_classifier_checkpoint(model, cp_name)
    model.to(device)
    check_acc(loader_test, model, device, name='test')


def load_fp32_init_for_qat(model):
    checkpoint = torch.load(
        checkpoint_path(opt.fp32_checkpoint_name), map_location='cpu')
    missing, unexpected = model.load_state_dict(
        checkpoint['state_dict'], strict=False)
    missing = [k for k in missing if not k.startswith('lutq_codebook.')]
    if unexpected:
        raise RuntimeError('unexpected FP32 checkpoint keys: %s' % unexpected)
    print('loaded FP32 init from %s' % checkpoint_path(opt.fp32_checkpoint_name))
    print('missing QAT-only keys: %d' % len(missing))
    model.epoch = 0
    model.lr_pre_epoch = []
    model.loss_pre_epoch = []
    model.testacc_pre_epoch = []


def qat_finetune(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    loader_train, loader_test = build_h5_loaders()
    device = get_device()

    model = make_qat_model()
    if opt.checkpoint_load_name:
        load_qat_checkpoint(model, opt.checkpoint_load_name)
    else:
        load_fp32_init_for_qat(model)
    model.to(device)

    check_acc(loader_test, model, device, name='initial 4W4A test')
    fit(model, loader_train, loader_test, device, opt.qat_max_epoch, opt.qat_lr,
        parse_milestones(opt.qat_milestones), 0, opt.qat_checkpoint_save_name)


def qat_test(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    _, loader_test = build_h5_loaders()
    device = get_device()

    model = make_qat_model()
    cp_name = opt.checkpoint_load_name or opt.qat_checkpoint_save_name
    load_qat_checkpoint(model, cp_name)
    model.to(device)
    check_acc(loader_test, model, device, name='4W4A test')


def real_quant_test(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    _, loader_test = build_h5_loaders()
    device = get_device()

    qat_model = make_qat_model()
    cp_name = opt.checkpoint_load_name or opt.qat_checkpoint_save_name
    load_qat_checkpoint(qat_model, cp_name)
    qat_model.eval()

    model = models.ALL_CNN_C_INT4(qat_model).to(device)
    check_acc(loader_test, model, device, name='real int4 W / int4 A test')


def int4_sim_test(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    _, loader_test = build_h5_loaders()

    qat_model = make_qat_model()
    cp_name = opt.checkpoint_load_name or opt.qat_checkpoint_save_name
    load_qat_checkpoint(qat_model, cp_name)
    qat_model.eval()

    model = ALLCNNInt4Simulator(qat_model).cpu()
    print('integer simulator code ranges:', model.code_ranges())
    check_acc(loader_test, model, torch.device('cpu'),
              name='int4 simulator test')


def pim_sim_test(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    _, loader_test = build_h5_loaders()

    qat_model = make_qat_model()
    cp_name = opt.checkpoint_load_name or opt.qat_checkpoint_save_name
    load_qat_checkpoint(qat_model, cp_name)
    qat_model.eval()

    model = ALLCNNPIMSimulator(
        qat_model, adc_bits=opt.pim_adc_bits, keep_rows=opt.pim_keep,
        stage11_keep_rows=opt.pim_stage11_keep,
        noise_sigma=opt.pim_noise_sigma).cpu()
    print('PIM simulator code ranges:', model.code_ranges())
    check_acc(loader_test, model, torch.device('cpu'),
              name='PIM simulator test', max_batches=opt.max_eval_batches)


def teacher_test(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    _, loader_test = build_h5_loaders()
    device = get_device()
    teacher = qkd.load_teacher(
        resolve_checkpoint_path(opt.teacher_checkpoint_path),
        num_classes=opt.num_classes, in_channels=opt.input_channels,
        arch=opt.teacher_arch, device=device)
    print('* * *  Checking accuracy on teacher test set')
    qkd.evaluate_teacher(loader_test, teacher, device)


def qkd_finetune(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    stage = opt.qkd_stage.upper()
    if stage not in ('SS', 'CS', 'TU'):
        raise ValueError('qkd_stage must be SS, CS, or TU')

    loader_train, loader_test = build_h5_loaders(
        return_index=(stage == 'TU' and opt.qkd_cache_teacher_logits))
    device = get_device()

    student = make_qat_model()
    if stage == 'SS':
        load_fp32_init_for_qat(student)
    else:
        student_checkpoint = qkd_default_student_checkpoint(stage)
        load_qat_checkpoint(student, student_checkpoint)
        print('loaded QKD student from %s'
              % resolve_checkpoint_path(student_checkpoint))
    student.to(device)

    teacher = None
    if stage in ('CS', 'TU'):
        teacher_path = opt.teacher_checkpoint_path
        if stage == 'TU':
            finetuned_teacher = checkpoint_path(opt.qkd_teacher_save_name)
            if os.path.exists(finetuned_teacher):
                teacher_path = finetuned_teacher
        teacher = qkd.load_teacher(
            resolve_checkpoint_path(teacher_path),
            num_classes=opt.num_classes, in_channels=opt.input_channels,
            arch=opt.teacher_arch, device=device)
    if stage == 'TU':
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    print('* * *  Initial QKD %s student accuracy' % stage)
    initial_student_acc = check_acc(loader_test, student, device, name='test')
    if teacher is not None:
        print('* * *  Initial QKD %s teacher accuracy' % stage)
        qkd.evaluate_teacher(loader_test, teacher, device)

    fit_qkd(stage, student, teacher, loader_train, loader_test, device,
            qkd_stage_save_name(stage), initial_student_acc)


if __name__ == '__main__':
    import fire
    fire.Fire()
