import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


D300_SIGNAL = np.array([
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
    10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0,
])
D300_NOISE = np.array([
    1.50, 1.67, 1.85, 2.02, 2.18, 2.34, 2.48, 2.61,
    2.76, 2.91, 3.06, 3.25, 3.42, 3.62, 3.85, 4.06,
    4.31, 4.52, 4.76, 5.01, 5.25, 5.50, 5.76, 6.01,
    6.29, 6.50, 6.83, 7.07,
])
D300_COEFFS = np.polyfit(D300_SIGNAL, D300_NOISE, 3)


def d300_noise(raw):
    max_value = max(float(raw.max()), 1e-10)
    clipped = torch.clamp(raw, min=1e-10, max=max_value)
    logsignal = 12.0 + torch.log2(clipped / max_value)
    c0, c1, c2, c3 = D300_COEFFS
    lognoise = ((c0 * logsignal + c1) * logsignal + c2) * logsignal + c3
    return torch.randn_like(clipped) * max_value * 2.0**(lognoise - 12.0)


class RawH5CIFARDataset(Dataset):
    """Read scan-generated CIFAR RAW/RGGB HDF5 files.

    Expected datasets:
      images: (N, 4, H, W) or (N, H, W, 4)
      labels: (N,)
      train: boolean split mask, True for train and False for test
    """

    def __init__(self, h5_path, train, augment=False, noise=False,
                 return_index=False):
        import h5py

        self.h5_path = str(h5_path)
        self.train = train
        self.augment = augment
        self.noise = noise
        self.return_index = return_index
        self._file = None

        with h5py.File(self.h5_path, 'r') as h5:
            for key in ('images', 'labels', 'train'):
                if key not in h5:
                    raise ValueError('%s missing required dataset %s' % (self.h5_path, key))

            shape = h5['images'].shape
            if len(shape) != 4:
                raise ValueError('expected 4D images in %s, got %s' % (self.h5_path, shape))
            if shape[1] == 4:
                self.channels_last = False
                self.image_shape = tuple(shape[1:])
            elif shape[-1] == 4:
                self.channels_last = True
                self.image_shape = (shape[-1], shape[1], shape[2])
            else:
                raise ValueError('expected RGGB images with 4 channels, got %s' % (shape,))

            self.dtype = h5['images'].dtype
            split = h5['train'][:].astype(bool)
            self.indices = np.where(split if train else ~split)[0]

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'] = None
        return state

    def _h5(self):
        if self._file is None:
            import h5py
            self._file = h5py.File(self.h5_path, 'r')
        return self._file

    def _augment(self, x):
        crop_h, crop_w = self.image_shape[1], self.image_shape[2]
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[2])
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        _, h, w = x.shape
        top = torch.randint(0, h - crop_h + 1, ()).item()
        left = torch.randint(0, w - crop_w + 1, ()).item()
        return x[:, top:top + crop_h, left:left + crop_w]

    def __getitem__(self, idx):
        h5 = self._h5()
        ds_idx = int(self.indices[idx])
        raw_np = h5['images'][ds_idx]
        if self.channels_last:
            raw_np = np.ascontiguousarray(raw_np.transpose(2, 0, 1))
        raw = torch.from_numpy(raw_np).float()
        if np.issubdtype(self.dtype, np.integer):
            raw = raw / float(np.iinfo(self.dtype).max)

        if self.noise:
            raw = torch.clamp(raw + d300_noise(raw), 0.0, 1.0)

        if self.augment:
            raw = self._augment(raw)

        label = int(h5['labels'][ds_idx])
        label = torch.tensor(label, dtype=torch.long)
        if self.return_index:
            return raw, label, torch.tensor(idx, dtype=torch.long)
        return raw, label
