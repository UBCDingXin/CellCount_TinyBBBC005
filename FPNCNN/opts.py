from argparse import ArgumentParser


def prepare_options():
    parser = ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--path_vgg_dataset', type=str, default='')
    parser.add_argument('--path_tinybbbc005', type=str, default='')
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--experiment_name', type=str, default='exp1')

    ''' Datast Settings '''
    parser.add_argument('--num_channels', type=int, default=1, metavar='N')
    parser.add_argument('--img_height', type=int, default=256, metavar='N')
    parser.add_argument('--img_width', type=int, default=256, metavar='N')
    parser.add_argument('--deleted_counts', type=str, default='None')

    ''' UNET Mask Settings '''
    parser.add_argument('--fpn_epochs', type=int, default=200)
    parser.add_argument('--fpn_resume_epoch', type=int, default=0)
    parser.add_argument('--fpn_batch_size_train', type=int, default=64)
    parser.add_argument('--fpn_lr_base', type=float, default=1e-3, help='base learning rate')
    parser.add_argument('--fpn_lr_decay_factor', type=float, default=0.01)
    parser.add_argument('--fpn_lr_decay_epochs', type=str, default='100_150', help='decay lr at which epoch; separate by _')
    parser.add_argument('--fpn_weight_decay', type=float, default=0)
    parser.add_argument('--fpn_transform', action='store_true', default=False, help='data augmentation')

    ''' DRCNN Settings '''
    parser.add_argument('--cnn_name', type=str, default="VGG19")
    parser.add_argument('--cnn_epochs', type=int, default=200)
    parser.add_argument('--cnn_resume_epoch', type=int, default=0)
    parser.add_argument('--cnn_batch_size_train', type=int, default=16)
    parser.add_argument('--cnn_batch_size_test', type=int, default=64)
    parser.add_argument('--cnn_lr_base', type=float, default=1e-2, help='base learning rate')
    parser.add_argument('--cnn_lr_decay_factor', type=float, default=1e-3)
    parser.add_argument('--cnn_lr_decay_epochs', type=str, default='100_150', help='decay lr at which epoch; separate by _')
    parser.add_argument('--cnn_weight_decay', type=float, default=0.1)
    parser.add_argument('--cnn_transform', action='store_true', default=False, help='data augmentation')


    args = parser.parse_args()

    return args
