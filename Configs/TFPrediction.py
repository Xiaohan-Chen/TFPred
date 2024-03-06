'''
Define hyperparameters.
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameters')

    # dataset
    parser.add_argument("--datadir", type=str, default="/home/datasets", help="data directory")
    parser.add_argument("--dataset", type=str, default="PU", choices=["PU"], help="log file path")
    parser.add_argument("--load", type=int, default=0, help="working condition")
    parser.add_argument("--num_train", type=int, default=500, help="the number of samples per class")
    parser.add_argument("--num_validation", type=int, default=100, help="the number of validation samples per class")
    parser.add_argument("--num_test", type=int, default=100, help="the number of test samples per class")
    parser.add_argument("--num_labels", type=int, default=30, help="the number of labeled samples per class")
    parser.add_argument("--ratio_labels", type=float, default=0.01, help="the ratio of labeld samples")
    parser.add_argument("--data_length", type=int, default=512, help="signal length of per sample")
    parser.add_argument("--labels", type=str, default="0,1,2,3,4,5,6,7,8,9", help="training classes")

    # pre-processing
    parser.add_argument("--window", type=int, default=384, help="time window, if not augment data, window=1024")
    parser.add_argument("--normalization", type=str, default="0-1", choices=["0-1"], help="normalization option")
    parser.add_argument("--backbone", type=str, default="ResNet1D", choices=["ResNet1D"])

    # training
    parser.add_argument("--mode", type=str, default="train", choices=["train", "tune", "train_then_tune", "evaluate"])
    parser.add_argument("--max_epochs", type=int, default=500, help="the number of maximum of training epoch")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='stepLR', choices=['stepLR'], help='the learning rate schedule')
    parser.add_argument("--num_workers", type=int, default=2, help="the number of dataloader workers")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])

    # tuning
    parser.add_argument("--tune_max_epochs", type=int, default=100, help="the number of maximum of traing epoch for linear evaluation")
    parser.add_argument("--backbone_lr", type=float, default=5e-3, help="learning rate for backbone")
    parser.add_argument("--classifier_lr", type=float, default=0.1, help="learning rate for classifier")

    # hyperparameters
    parser.add_argument('--gamma', type=float, default=0.8, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=int, default=60, help='the learning rate decay for step and stepLR')

    args = parser.parse_args()

    return args