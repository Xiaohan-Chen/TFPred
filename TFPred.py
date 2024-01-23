"""
Author: Xiaohan Chen
Email: cxh_bb@outlook.com
"""

import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import *

import Configs.TFPrediction as parms
from Preparedata import PU
from Datasets import Dataset
from Models import ResNet1D
from Losses.CrossCorrelation import CrossCorrelationLoss
from Utils import utils
from Utils.logger import setlogger

def load_data(args):
    datadict = PU.PUloader(args)
    
    # shuffle the datasets
    np.random.seed(28)
    datadict = {key:np.random.permutation(datadict[key]) for key in datadict.keys()}

    # split the dataset
    train_datadict = {key:datadict[key][:args.num_train] for key in datadict.keys()}
    val_datadict = {key:datadict[key][args.num_train : args.num_train + args.num_validation] for key in datadict.keys()}
    test_datadict = {key:datadict[key][-args.num_test:] for key in datadict.keys()}

    # creat datasets
    train_dataset = Dataset.AugmentDasetsetTFPair(train_datadict)
    evaluate_dataset = Dataset.BaseDataset(train_datadict)
    val_dataset = Dataset.BaseDataset(val_datadict)
    test_dataset = Dataset.BaseDataset(test_datadict)

    labeled_indices, _ = Dataset.relabel_dataset(args, evaluate_dataset)
    sampler = torch.utils.data.SubsetRandomSampler(labeled_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,\
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    evaluate_loader = DataLoader(evaluate_dataset, batch_size=args.batch_size, sampler=sampler,\
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, \
        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, \
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return train_loader, evaluate_loader, val_loader, test_loader

# ===== Define encoder =====
class ModelBase(nn.Module):
    '''
    Encoder
    '''
    def __init__(self, dim=128) -> None:
        super().__init__()

        self.net = ResNet1D.resnet18(norm_layer=None)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, dim)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        x = self.flatten(x)
        x = self.fc(x)
        return x

class TFPrediction(nn.Module):
    def __init__(self, dim=128) -> None:
        super().__init__()
        self.encoderT = ModelBase()
        self.encoderF = ModelBase()

        self.PredictionF = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, dim),
        )


    def forward(self, x_t, x_f):
        x_t = self.encoderT(x_t)
        x_f = self.encoderF(x_f)

        x_f = self.PredictionF(x_f)

        return x_t, x_f

def test_evaluate(args, model, dataloader, criterion, device):
    model.eval()
    lossmeter = utils.AverageMeter("test_loss")
    accmeter = utils.AverageMeter("test_acc")

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y.long())
            acc = utils.accuracy(output, y)

            lossmeter.update(loss.item())
            accmeter.update(acc)

    return accmeter.avg, lossmeter.avg

def train_evaluate(args, model, dataloader, optimizer, criterion, device):
    model.train()
    lossmeter = utils.AverageMeter("train_loss")
    accmeter = utils.AverageMeter("train_acc")

    with tqdm(total=len(dataloader), ncols=70, leave=False) as pbar:
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y.long())
            acc = utils.accuracy(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossmeter.update(loss.item())
            accmeter.update(acc)

            pbar.update()

    return accmeter.avg, lossmeter.avg

def main_evaluate(args):
    # Using GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, evaluate_loader, val_loader, test_loader = load_data(args)

    classes = len(args.labels.split(","))
    model = ModelBase(dim=classes).to(device)
    checkpoint = torch.load("./History/TFPred_checkpoint.pth", map_location="cpu")
    for k in list(checkpoint.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('encoderT'):
            # remove prefix
            checkpoint[k[len("encoderT."):]] = checkpoint[k]
        # delete renamed or unused k
        del checkpoint[k]
    for k in list(checkpoint.keys()):
        if k.startswith('fc'):
            del checkpoint[k]
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    assert missing_keys == ["fc.weight", "fc.bias"]
    model.fc.weight.data.normal_(mean=0.0, std=0.1)
    model.fc.bias.data.zero_()

    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)
    
    criterion = nn.CrossEntropyLoss().cuda()
    param_groups = [
        dict(params=classifier_parameters, lr=args.classifier_lr),
        dict(params=model_parameters, lr=args.backbone_lr)
    ]
    optimizer = torch.optim.SGD(param_groups, 0, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.tune_max_epochs)

    best_acc = 0.0
    logging.info(">>>>> TFPred Semi-Supervised Evaluation ...")
    for epoch in range(args.tune_max_epochs):
        train_acc, train_loss = train_evaluate(args, model, evaluate_loader, optimizer, criterion, device)
        val_acc, val_loss = test_evaluate(args, model, val_loader, criterion, device)

        lr_scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            test_acc, _ = test_evaluate(args, model, test_loader, criterion, device)
        
        logging.info(f"Epoch: {epoch+1}/{args.tune_max_epochs}, train loss: {train_loss:.4f}, "
        f"train_acc: {train_acc:6.2f}%, val loss: {val_loss:.4f}, val_acc: {val_acc:6.2f}%")
    logging.info(f"Best val acc: {best_acc:6.2f}%, test acc: {test_acc:6.2f}%")
    logging.info("="*15+"TFPred Evaluation Done!"+"="*15)

def train(args, model, train_loader, criterion, optimizer, device):
    model.train()

    lossmeter = utils.AverageMeter("train_loss")

    with tqdm(total=len(train_loader), ncols=70, leave=False) as pbar:
        for i, (x_t, x_f, _) in enumerate(train_loader):
            x_t, x_f = x_t.to(device), x_f.to(device)

            x_t, x_f = model(x_t, x_f)
            loss = criterion(x_t, x_f)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossmeter.update(loss.item())
        
            pbar.update()

    return lossmeter.avg

def main(args):
    # Using GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, _, _, _ = load_data(args)

    # load model
    model = TFPrediction().to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs)

    # loss
    criterion = CrossCorrelationLoss()
    
    logging.info(">>>>> TFPred Pre-training ...")
    best_loss = 1e9
    for epoch in range(args.max_epochs):
        train_loss = train(args, model, train_loader, criterion, optimizer, device)

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "./History/TFPred_checkpoint.pth")
        
        logging.info(f"Epoch: {epoch+1:>3}/{args.max_epochs}, train_loss: {train_loss:.4f}, "
                f"current lr: {lr_scheduler.get_last_lr()[0]:.6f}")
    logging.info("="*15+"TFPred Pre-training Done!"+"="*15)

if __name__ == "__main__":

    args = parms.parse_args()

    if not os.path.exists("./History"):
        os.makedirs("./History")

    # set the logger
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    setlogger("./logs/TFPred.log")

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    if args.mode  == "train":
        main(args)
    elif args.mode == "tune":
        main_evaluate(args)
    elif args.mode == "train_then_tune":
        main(args)
        main_evaluate(args)
