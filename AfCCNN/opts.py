from argparse import ArgumentParser


def prepare_options():
    parser = ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--path_tinybbbc005', type=str, default='')
    parser.add_argument('--seed', type=int, default=2020, metavar='S', help='random seed')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--experiment_name', type=str, default='exp1',
                        choices=['exp1',
                                 'exp2_rd1','exp2_rd2','exp2_rd3',
                                 'exp3_rd1','exp3_rd2','exp3_rd3'])

    ''' Datast Settings '''
    parser.add_argument('--num_channels', type=int, default=1, metavar='N')
    parser.add_argument('--img_height', type=int, default=256, metavar='N')
    parser.add_argument('--img_width', type=int, default=256, metavar='N')

    ''' CNN Settings '''
    parser.add_argument('--cnn_name', type=str, default='ResNet34',
                        choices=['ResNet18', 'ResNet34', 'ResNet50', 'VGG11', 'VGG13', 'VGG16', 'VGG19'])
    parser.add_argument('--predtype', type=str, default='class', choices=['class', 'regre'],
                        help='Prediction type for the cnn: classication or regression;')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--batch_size_train', type=int, default=16)
    parser.add_argument('--batch_size_test', type=int, default=64)
    parser.add_argument('--lr_base', type=float, default=1e-2, help='base learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='100_150', help='decay lr at which epoch; separate by _')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--transform', action='store_true', default=False, help='conventional data augmentation')

    ''' Data Augmentation by Overlaying Multiple Images '''
    parser.add_argument('--dataAugment', action='store_true', default=False,
                        help='Do data augmentation?')
    parser.add_argument('--nfake', type=int, default=20,
                        help='number of fake images for each stain and blur combination') #25*5/6~= 20
    parser.add_argument('--da_flip', action='store_true', default=False,
                        help='Do flipping in data augmentation?')
    parser.add_argument('--da_filter', action='store_true', default=False,
                        help='Do filtering in DA?')
    parser.add_argument('--one_formula_per_class', action='store_true', default=False,
                        help='One formula for each cell count?')

    ''' Ensemble '''
    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='Ensemble a classification CNN and LQReg?')
    parser.add_argument('--ensemble_regre_model', type=str, default='LQReg',
                        choices=['LQReg', 'ResNet18', 'ResNet34', 'ResNet50', 'VGG11', 'VGG13', 'VGG16', 'VGG19'],
                        help='Ensemble the class CNN with which regression model?')
    parser.add_argument('--ensemble_regre_cnn_path', type=str, default='')

    args = parser.parse_args()

    return args
