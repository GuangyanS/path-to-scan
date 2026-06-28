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


def train(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    loader_train, loader_test = build_h5_loaders()

    if opt.use_gpu == True and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = getattr(models, opt.model)(
        num_classes=opt.num_classes, in_channels=opt.input_channels)
    if opt.use_trained_model == True:
        cp_name = opt.model
        if opt.checkpoint_load_name != None:
            cp_name = opt.checkpoint_load_name
        model.load(opt.test_model_path + cp_name)
    model.to(device)

    lr = opt.lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=opt.weight_decay, nesterov=True)

    if isinstance(opt.milestones, (tuple, list)):
        milestones = [int(x) for x in opt.milestones]
    else:
        milestones = [int(x) for x in str(opt.milestones).split(',') if x]
    if opt.warmup > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=opt.warmup)
        main_sched = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, main_sched],
            milestones=[opt.warmup])
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)

    best_testacc = 0.0
    for epoch in range(opt.max_epoch):
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
                      .format(epoch + 1, opt.max_epoch, loss_now,
                              optimizer.param_groups[0]['lr']))

        scheduler.step()

        if (epoch + 1) % opt.eval_every == 0:
            testacc = check_acc(loader_test, model, device, name='test')
            model.update_epoch(optimizer.param_groups[0]['lr'], loss_now, testacc)

            if testacc > best_testacc:
                best_testacc = testacc
                saved = model.save(opt.checkpoint_save_name)
                print('**  New best test acc %.2f%% -> saved %s' % (testacc * 100, saved))


def test(**kwargs):
    opt.parse(kwargs)
    set_seed(opt.seed)
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    _, loader_test = build_h5_loaders()

    if opt.use_gpu == True and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = getattr(models, opt.model)(
        num_classes=opt.num_classes, in_channels=opt.input_channels)

    cp_name = opt.checkpoint_load_name or opt.checkpoint_save_name or opt.model
    model.load(opt.test_model_path + cp_name)
    model.to(device)
    check_acc(loader_test, model, device, name='test')


if __name__ == '__main__':
    import fire
    fire.Fire()
