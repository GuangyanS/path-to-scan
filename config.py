import warnings

class DefaultConfig(object):

    model = 'ALL_CNN_C'
    num_classes = 100
    input_channels = 4
    seed = 0
    checkpoint_load_name = None
    checkpoint_load_strict = True
    checkpoint_save_name = 'ALL_CNN_C_c100_rggb_h5_bn_refstyle'
    fp32_checkpoint_name = 'ALL_CNN_C_c100_rggb_h5_bn_refstyle'
    qat_checkpoint_save_name = 'ALL_CNN_C_c100_rggb_h5_w4a4_qat'
    pcn_cycles = 3
    pcn_alpha_init = 0.000001
    pcn_init_extras_from_ff = True
    teacher_checkpoint_path = 'b4_100'
    teacher_arch = 'efficientnet_v2_l'
    qkd_stage = 'SS'
    qkd_student_checkpoint_name = None
    qkd_checkpoint_save_name = 'ALL_CNN_C_c100_rggb_h5_w4a4_qkd'
    qkd_teacher_save_name = 'efficientnetv2_l_qkd_teacher'
    qkd_use_lutq = False
    lutq_group_size = 8
    lutq_int_max = 7
    input_bits = 16
    use_pact = False
    pact_alpha = 6.0
    qkd_cache_teacher_logits = False

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
    qkd_max_epoch = None
    qkd_lr = None
    qkd_milestones = None
    qkd_temperature = None
    qkd_kd_weight = 0.3
    qkd_loss = 'kl'
    dkd_alpha = 1.0
    dkd_beta = 2.0
    qkd_teacher_lr_factor = 0.01
    qkd_grad_clip = 1.0


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
