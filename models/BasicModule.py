import os
import torch


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

        self.epoch = 0
        self.lr_pre_epoch = []
        self.loss_pre_epoch = []
        self.testacc_pre_epoch = []

    def load(self, path, strict=True):
        # tolerate names with or without the .pth suffix
        if not os.path.exists(path) and os.path.exists(path + '.pth'):
            path = path + '.pth'
        checkpoint = torch.load(path, map_location='cpu')
        self.epoch = checkpoint['epoch']
        self.lr_pre_epoch = checkpoint['lr']
        self.loss_pre_epoch = checkpoint['loss']
        self.testacc_pre_epoch = checkpoint['testacc']
        incompatible = self.load_state_dict(checkpoint['state_dict'],
                                            strict=strict)
        if not strict:
            print('loaded %s with missing keys: %d, unexpected keys: %d'
                  % (path, len(incompatible.missing_keys),
                     len(incompatible.unexpected_keys)))
        return incompatible

    def save(self, name=None):
        if name is None:
            name = 'checkpoints/' + self.model_name
        else:
            name = 'checkpoints/' + name

        if not name.endswith('.pth'):
            name += '.pth'

        dirname = os.path.dirname(name)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        torch.save({
            'epoch': self.epoch,
            'lr': self.lr_pre_epoch,
            'loss': self.loss_pre_epoch,
            'testacc': self.testacc_pre_epoch,
            'state_dict': self.state_dict(),
        }, name)

        return name

    def update_epoch(self, lr_, loss_, testacc_):
        self.epoch += 1
        self.lr_pre_epoch.append(lr_)
        self.loss_pre_epoch.append(loss_)
        self.testacc_pre_epoch.append(testacc_)
        assert len(self.lr_pre_epoch) == len(self.loss_pre_epoch) == len(self.testacc_pre_epoch)

    def epoches(self):
        return len(self.lr_pre_epoch)
