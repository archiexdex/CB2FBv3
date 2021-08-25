def add_arguments(parser):
    '''
    Add your arguments here if needed.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''

    parser.add_argument('--no',
                        type=int,
                        help='the training model number(name)')
    parser.add_argument('-y', '--yes',
                        action='store_true',
                        help='if true, override folders')
    parser.add_argument('--total_epoch',
                        type=int,
                        default=200,
                        help='total training epoch')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='learning rate')
    parser.add_argument('--seed',
                        type=int,
                        default=9487,
                        help='The random seed number')
    parser.add_argument('--no_crop',
                        action='store_true',
                        help='if set true, do not crop image')
    parser.add_argument('--crop_size',
                        type=int,
                        default=128,
                        help='the size to crop image')
    parser.add_argument('--no_early_stop',
                        action='store_true',
                        help='if set true, do not early stop training')
    parser.add_argument('--patience',
                        type=int,
                        default=10,
                        help='the epoch patience for early stoping ')
    parser.add_argument('--load_mode',
                        type=str,
                        default='best',
                        choices=["best", "freq"],
                        help='differnt mode of saving model')
    parser.add_argument('--warm_epoch',
                        type=int,
                        default=10,
                        help='before the setting number, train model as pure gan instead of cycle gan')

    parser.add_argument('--sample_mode',
                        type=str,
                        default="fix",
                        choices=["all", "random", "fix"],
                        help='the type of sampling validation dataset')

    parser.add_argument('--train_data_path',
                        type=str,
                        default='data/train_data',
                        help='path for saving training data')
    parser.add_argument('--test_data_path',
                        type=str,
                        default='data/test_data',
                        help='path for saving testing data')
    parser.add_argument('--npy_data_path',
                        type=str,
                        default='data/npy_data',
                        help='path for npy data')
    parser.add_argument('--mask_data_path',
                        type=str,
                        default='data/mask_data',
                        help='path for mask data')
    parser.add_argument('--png_data_path',
                        type=str,
                        default='data/png_data',
                        help='path for png data')
    parser.add_argument('--cpt_dir',
                        type=str,
                        default='checkpoints',
                        help='path for saving training checkpoints')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='path for saving training logs')
    parser.add_argument('--cfg_dir',
                        type=str,
                        default='configs',
                        help='path for saving training config parameters')
    parser.add_argument('--rst_dir',
                        type=str,
                        default='results',
                        help='path for saving results')

    parser.add_argument('--debug',
                        action='store_true',
                        help='if true, enable debug mode')

    return parser
