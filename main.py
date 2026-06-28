import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import opt
from h5_dataset import RawH5CIFARDataset
import models
import qkd
from simulator import ALLCNNInt4Simulator


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


def check_acc(loader, model, device, name='test'):
    print('* * *  Checking accuracy on %s set' % name)
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum().item()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('****    Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def build_h5_loaders():
    train_set = RawH5CIFARDataset(
        opt.h5_path, train=True, augment=opt.raw_augment,
        noise=opt.raw_noise)
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
        'SS': {'epochs': 30, 'lr': opt.qat_lr, 'milestones': '10,20',
               'temperature': 2.0},
        'CS': {'epochs': 100, 'lr': opt.qat_lr, 'milestones': '80,120',
               'temperature': 2.0},
        'TU': {'epochs': 70, 'lr': opt.qat_lr * 0.5, 'milestones': '40,60',
               'temperature': 2.0},
    }
    if stage not in defaults:
        raise ValueError('qkd_stage must be SS, CS, or TU')
    return defaults[stage]


def qkd_default_student_checkpoint(stage):
    if opt.qkd_student_checkpoint_name:
        return opt.qkd_student_checkpoint_name
    if stage == 'CS':
        return opt.qkd_checkpoint_save_name + '_ss'
    if stage == 'TU':
        return opt.qkd_checkpoint_save_name + '_cs'
    return opt.qat_checkpoint_save_name


def qkd_stage_save_name(stage):
    return opt.qkd_checkpoint_save_name + '_' + stage.lower()


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
    scheduler = make_scheduler(optimizer, milestones, 0)

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
    for epoch in range(max_epoch):
        student.train()
        if teacher is not None:
            teacher.train(stage == 'CS')

        loss_now = 0.0
        for ii, (data, label) in tqdm(enumerate(loader_train)):
            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            student_scores = student(data)
            student_ce = F.cross_entropy(student_scores, label)
            loss = student_ce

            if teacher is not None:
                t_data = qkd.prepare_teacher_inputs(data)
                if stage == 'CS':
                    teacher_scores = teacher(t_data)
                    student_kd = qkd.soft_kl_loss(
                        student_scores, teacher_scores.detach(),
                        temperature)
                    teacher_ce = F.cross_entropy(teacher_scores, label)
                    teacher_kd = qkd.soft_kl_loss(
                        teacher_scores, student_scores.detach(),
                        temperature)
                    loss = (student_ce + opt.qkd_kd_weight * student_kd
                            + teacher_ce + opt.qkd_kd_weight * teacher_kd)
                else:
                    with torch.no_grad():
                        teacher_scores = teacher(t_data)
                    student_kd = qkd.soft_kl_loss(
                        student_scores, teacher_scores, temperature)
                    loss = student_ce + opt.qkd_kd_weight * student_kd

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

    model = models.ALL_CNN_C_QAT(
        num_classes=opt.num_classes, in_channels=opt.input_channels)
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

    model = models.ALL_CNN_C_QAT(
        num_classes=opt.num_classes, in_channels=opt.input_channels)
    cp_name = opt.checkpoint_load_name or opt.qat_checkpoint_save_name
    model.load(opt.test_model_path + cp_name)
    model.to(device)
    check_acc(loader_test, model, device, name='4W4A test')


def real_quant_test(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    _, loader_test = build_h5_loaders()
    device = get_device()

    qat_model = models.ALL_CNN_C_QAT(
        num_classes=opt.num_classes, in_channels=opt.input_channels)
    cp_name = opt.checkpoint_load_name or opt.qat_checkpoint_save_name
    qat_model.load(opt.test_model_path + cp_name)
    qat_model.eval()

    model = models.ALL_CNN_C_INT4(qat_model).to(device)
    check_acc(loader_test, model, device, name='real int4 W / int4 A test')


def int4_sim_test(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    _, loader_test = build_h5_loaders()

    qat_model = models.ALL_CNN_C_QAT(
        num_classes=opt.num_classes, in_channels=opt.input_channels)
    cp_name = opt.checkpoint_load_name or opt.qat_checkpoint_save_name
    qat_model.load(opt.test_model_path + cp_name)
    qat_model.eval()

    model = ALLCNNInt4Simulator(qat_model).cpu()
    print('integer simulator code ranges:', model.code_ranges())
    check_acc(loader_test, model, torch.device('cpu'),
              name='int4 simulator test')


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

    loader_train, loader_test = build_h5_loaders()
    device = get_device()

    student = models.ALL_CNN_C_QAT(
        num_classes=opt.num_classes, in_channels=opt.input_channels)
    if stage == 'SS':
        load_fp32_init_for_qat(student)
    else:
        student_checkpoint = qkd_default_student_checkpoint(stage)
        student.load(resolve_checkpoint_path(student_checkpoint))
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
