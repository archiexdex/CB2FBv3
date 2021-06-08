def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''

    parser.add_argument('--no', type=int, help='')
    parser.add_argument('--total_epoch', type=int, default=200, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='')
    parser.add_argument('--seed', type=int, default=9487, help='')
    parser.add_argument('--crop', action='store_true', default=True, help='')
    parser.add_argument('--crop_size', type=int, default=128, help='')
    parser.add_argument('--patience', type=int, default=10, help='')
    parser.add_argument('--early_stop', action='store_true',  help='')
    parser.add_argument('-y', '--yes', action='store_true', help='')
    parser.add_argument('--load_mode', type=str, default='best', help='best or freq')

    parser.add_argument('--train_data_path', type=str, default='data/train_data', help='')
    parser.add_argument('--test_data_path',  type=str, default='data/test_data', help='')
    parser.add_argument('--npy_data_path',   type=str, default='data/npy_data', help='')
    parser.add_argument('--mask_data_path',  type=str, default='data/mask_data', help='')
    parser.add_argument('--png_data_path',   type=str, default='data/png_data', help='')
    parser.add_argument('--cpt_dir', type=str, default='checkpoints', help='')
    parser.add_argument('--log_dir', type=str, default='logs', help='')
    parser.add_argument('--cfg_dir', type=str, default='configs', help='')
    parser.add_argument('--rst_dir', type=str, default='results', help='')

    parser.add_argument('--debug', action='store_true', default=False, help='')
    #parser.add_argument('--', type=, default=, help='')
    #parser.add_argument('--', type=, default=, help='')
    #parser.add_argument('--', type=, default=, help='')

    return parser
