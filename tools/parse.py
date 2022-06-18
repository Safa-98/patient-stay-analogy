import os
import argparse

parser = argparse.ArgumentParser(description='help')

parser.add_argument(
        '--data-dir',
        type=str,
        default='processed_id',
        help='selected and preprocessed data directory'
        )

# method settings
parser.add_argument(
        '--split-num',
        metavar='split num',
        type=int,
        default=4000,
        help='split num'
        )

# model parameters
parser.add_argument(
        '--embed-size',
        metavar='EMBED SIZE',
        type=int,
        default=200,
        help='embed size'
        )


# traing process setting
parser.add_argument('--phase',
        default='train',
        type=str,
        help='train/test phase')
parser.add_argument(
        '--batch-size',
        '-b',
        metavar='BATCH SIZE',
        type=int,
        default=64,
        help='batch size'
        )
parser.add_argument('--model-path', type=str, default='processed_id/models/best.ckpt', help='model path')
parser.add_argument(
        '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 32)')
parser.add_argument('--lr',
        '--learning-rate',
        default=0.0001,
        type=float,
        metavar='LR',
        help='initial learning rate')
parser.add_argument('--epochs',
        default=50,
        type=int,
        metavar='N',
        help='number of total epochs to run')

args = parser.parse_args()

args.data_dir = 'processed_id'
args.files_dir = os.path.join(args.data_dir, 'files')
args.resample_dir = os.path.join(args.data_dir, 'resample_data')
args.initial_dir = os.path.join(args.data_dir, 'initial_data')
