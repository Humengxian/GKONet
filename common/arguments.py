import numpy as np
import random
import argparse
import torch
import os
import pprint
import logging
import colorlog

def create_logger(logdir):
    os.makedirs(logdir, exist_ok=True)
    log_file = os.path.join(logdir, 'log.txt')
    log_colors_config = {
    'DEBUG': 'white', 
    'INFO': 'white',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
    }

    logger = logging.getLogger('logger_name')
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=log_file, mode='a', encoding='utf8')
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter(
        fmt='[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S'
    )
    console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s%(message)s',
        datefmt='%Y-%m-%d  %H:%M:%S',
        log_colors=log_colors_config
    )
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    console_handler.close()
    file_handler.close()

    return logger

def parse_args(is_3dhp=False):
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, help='target dataset') # h36m or humaneva
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, help='2D detections to use')
    parser.add_argument('--subjects-train', default='S1,S5,S6,S7,S8', type=str, help='training subjects separated by comma')
    parser.add_argument('--subjects-test', default='S9,S11', type=str, help='test subjects separated by comma')
    parser.add_argument('--actions', default='*', type=str, help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('-g', '--gpu', default=6, type=int, help='gpu id')
    parser.add_argument('-exp', '--experiment', default='test', type=str, help='experiment name')

    # Model arguments
    parser.add_argument('-nj', '--num_joints', default=17, type=int, help='number of joints') 
    parser.add_argument('-njd', '--num_joint_dim', default=5, type=int, help='number of joints') 
    parser.add_argument('-pd', '--pose_dim', default=32, type=int, help='first branch hidden dimension')
    parser.add_argument('-jd', '--joint_dim', default=128, type=int, help='second branch dimension')
    parser.add_argument('--depth', default=6, type=int, help='block number')

    # Train arguments num_workers
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
    parser.add_argument('-seed', '--seed', default=1, type=int, help='batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=30, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batchsize', default=512, type=int, help='batch size in terms of predicted frames')
    parser.add_argument('-mdp', '--mlp_drop_pose', default=0., type=float, help='dropout probability')
    parser.add_argument('-adp', '--attn_drop_pose', default=0., type=float, help='dropout probability')
    parser.add_argument('-mdj', '--mlp_drop_joint', default=0., type=float, help='dropout probability')
    parser.add_argument('-adj', '--attn_drop_joint', default=0., type=float, help='dropout probability')
    parser.add_argument('-droppath', '--drop_path_rate', default=0.5, type=float, help='dropout probability of path')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('-wd', '--weight_decay', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('-lrd', '--lr_decay', default=0.99, type=float, help='initial learning rate')
    parser.add_argument('-frame', '--number-of-frames', default='1', type=int, help='how many frames used as input')

    # Experimental
    parser.add_argument('-s', '--stride', default=1, type=int, help='chunk size to use during training')
    parser.add_argument('--subset', default=1, type=float, help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--eval', default=False, type=bool, help='train or eval')
    parser.add_argument('--checkpoint', default='./', type=str, help='pre-train model save path')
    
    parser.set_defaults(data_augmentation=True)

    args = parser.parse_args()
    
    if is_3dhp:
        args.dataset = '3dhp'
        args.keypoints = 'gt'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

    if args.eval:
        args.experiment = args.keypoints + '_' + args.dataset + '_eval_' + args.experiment
    else:
        args.experiment = args.keypoints + '_' + args.dataset + '_train_' + args.experiment
    experiments_dir = os.path.join('experiment', args.experiment)    
    logger = create_logger(experiments_dir)

    if args.seed == 1:
        seed = random.randint(1, 1000000)
    else:
        seed = args.seed
        
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(experiments_dir, exist_ok=True)
    logger.info('====================================================================')
    for k in list(vars(args).keys()):
        logger.info('(Args) %s: %s' % (k, vars(args)[k]))
    logger.info('(Args) %s: %d' % ('seed', seed))
    return args, logger