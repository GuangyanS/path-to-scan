import warnings

class DefaultConfig(object):

    model = 'ALL_CNN_C'
    num_classes = 100
    input_channels = 4
    seed = 0
    checkpoint_load_name = None
    checkpoint_save_name = 'ALL_CNN_C_c100_rggb_h5_bn_refstyle'
    fp32_checkpoint_name = 'ALL_CNN_C_c100_rggb_h5_bn_refstyle'
    qat_checkpoint_save_name = 'ALL_CNN_C_c100_rggb_h5_w4a4_qat'

    test_model_path = './checkpoints/'
    use_trained_model = False

    h5_path = './datasets/cifar100_raw.h5'
    raw_augment = True
    raw_noise = True

    disable_cudnn = True
    milestones = '100,150'
    warmup = 5
    eval_every = 1


# config of training 1
    batch_size = 128
    use_gpu = True
    num_workers = 2
    print_freq = 5000
    debug_mode = True

# config of training 2
    max_epoch = 200
    lr = 0.1
    weight_decay = 0.0001
    qat_max_epoch = 50
    qat_lr = 0.001
    qat_milestones = '30,45'


def parse(self, kwargs):
# update the config according to kwargs
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__') and not callable(v):
            print(k, getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()
