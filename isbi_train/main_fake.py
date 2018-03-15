#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import time
import os
import pandas as pd
from preparedata_fake import prepare_trainval_data
from preparetestdata import prepare_test_data
import argparse
from evaluate_utils import plot_auc, calculate_auc
from visdom import Visdom

TopK = 5
ConCatCount = 3
FeatCount = 128


def _notice(epoch):
    print "*" * 30  + str(epoch) + "*" * 30

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.dropout = nn.Dropout(0.7)
        self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
        self.Relu = nn.ReLU()

    def forward(self, centerFeat):
        # batch * TopK * concat * feat
        featSize = centerFeat.size()
        out = self.pool(centerFeat)
        out = self.dropout(out)
        out = self.Relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = out.view(featSize[0], featSize[1])
        base_prob = torch.sigmoid(self.baseline)

        valid_nod_prob = torch.prod(1 - out, dim = 1)
        dummy_nod_prob = (1 - base_prob.expand(out.size()[0]))
        casePred = 1 - valid_nod_prob * dummy_nod_prob
        # print "otuput {}".format(out)
        return casePred, out


class FCDataSet(Dataset):
    def __init__(self, X, Y, Diameter=None, Volume=None, phase="train"):
        self.X = X
        self.Y = Y
        self.phase = phase

        if self.phase == "test":
            self.Diameter = Diameter
            self.Volume = Volume

    def __getitem__(self, idx):
        # note, in testing mode, the self.Y actully returns the label name
        if self.phase == "test":
            return self.X[idx, :], self.Y[idx], self.Diameter[idx, :], self.Volume[idx, :]
        else:
            return self.X[idx, :], self.Y[idx]

    def __len__(self):
        # actually returns the 1st dim for X
        return len(self.Y)


def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        if self.size_average:
            return torch.mean(-torch.log(inputs) * targets * torch.pow(1 - inputs, self.gamma) -
                              torch.log(1 - inputs) * (1 - targets) * torch.pow(inputs, self.gamma))
        else:
            return torch.sum(-torch.log(inputs) * targets * torch.pow(1 - inputs, self.gamma) -
                             torch.log(1 - inputs) * (1 - targets) * torch.pow(inputs, self.gamma))


class FCLoss(nn.Module):
    def __init__(self):
        super(FCLoss, self).__init__()
        self.num_hard = 0
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()

    def forward(self, output, labels, train=True):
        batch_size = labels.size(0)

        pos_idcs = labels > 0.5
        pos_output = output[pos_idcs]
        pos_labels = labels[pos_idcs]

        neg_idcs = labels <= 0.5
        neg_output = output[neg_idcs]
        neg_labels = labels[neg_idcs]

        if self.num_hard > 0 and train:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)

        if len(pos_output) > 0:
            # pos_prob = self.sigmoid(pos_output)
            pos_prob = pos_output

            if len(neg_output) > 0:
                # neg_prob = self.sigmoid(neg_output)
                neg_prob = neg_output
                classify_loss = 0.5 * self.classify_loss(
                    pos_prob, pos_labels) + 0.5 * self.classify_loss(
                    neg_prob, neg_labels)
                neg_correct = (neg_prob.data < 0.5).sum()
                neg_total = len(neg_prob)
            else:
                classify_loss = 0.5 * self.classify_loss(
                    pos_prob, pos_labels)
                neg_correct = 0
                neg_total = 0

            pos_correct = (pos_prob.data >= 0.5).sum()
            pos_total = len(pos_prob)
        else:
            if len(neg_output) > 0:
                # neg_prob = self.sigmoid(neg_output)
                neg_prob = neg_output

                classify_loss = 0.5 * self.classify_loss(
                    neg_prob, neg_labels)
                neg_correct = (neg_prob.data < 0.5).sum()
                neg_total = len(neg_prob)
            else:
                neg_correct = 0
                neg_total = 0
                print ("warining ! neither neg nor pos samples feeded, check the data")
            pos_correct = 0
            pos_total = 0

        classify_loss_data = classify_loss.data[0]
        return classify_loss, classify_loss_data, pos_correct, pos_total, neg_correct, neg_total


def FCTrain(data_loader, net, loss, epoch, optimizer, get_lr):
    start_time = time.time()

    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    loss_epoch, pos_c_epoch, pos_t_epoch, neg_c_epoch, neg_t_epoch = 0, 0, 0, 0, 0
    Yval_t, Ypred_t = [], []
    for i, (data, target) in enumerate(data_loader):
        data = Variable(data.cuda(async=True))
        target = Variable(target.cuda(async=True))

        casePred, nodPred = net(data)
        loss_output, loss_data, \
        pos_correct, pos_total, neg_correct, neg_total \
            = loss(casePred, target)
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()

        loss_epoch += loss_data
        pos_c_epoch += pos_correct
        neg_c_epoch += neg_correct
        pos_t_epoch += pos_total
        neg_t_epoch += neg_total

        Ypred_t.extend(list(casePred.data.cpu().numpy()))
        Yval_t.extend(list(target.data.cpu().numpy()))

    fpr, tpr, auc = calculate_auc(Yval_t, Ypred_t)
    print "train: loss {:.4f}, POS {:.4f}, NEG {:.4f}, AUC {:.4f}".\
        format(loss_epoch, float(pos_c_epoch) / pos_t_epoch, float(neg_c_epoch) / neg_t_epoch, auc)
    end_time = time.time()

    return loss_epoch, float(pos_c_epoch) / pos_t_epoch, float(neg_c_epoch) / neg_t_epoch, auc



def FCValidate(data_loader, net, loss):
    start_time = time.time()
    net.eval()

    loss_epoch, pos_c_epoch, pos_t_epoch, neg_c_epoch, neg_t_epoch = 0, 0, 0, 0, 0
    Yval_t, Ypred_t = [], []
    for i, (data, target) in enumerate(data_loader):
        data = Variable(data.cuda(async=True), volatile=True)
        target = Variable(target.cuda(async=True), volatile=True)

        casePred, nodPred = net(data)
        loss_output, loss_data, \
        pos_correct, pos_total, neg_correct, neg_total \
            = loss(casePred, target, train=False)

        loss_epoch += loss_data
        pos_c_epoch += pos_correct
        neg_c_epoch += neg_correct
        pos_t_epoch += pos_total
        neg_t_epoch += neg_total

        Ypred_t.extend(list(casePred.data.cpu().numpy()))
        Yval_t.extend(list(target.data.cpu().numpy()))

    fpr, tpr, auc = calculate_auc(Yval_t, Ypred_t)
    print "val: loss {:.4f}, POS {:.4f}, NEG {:.4f}, AUC {:.4f}".\
        format(loss_epoch, float(pos_c_epoch) / pos_t_epoch, float(neg_c_epoch) / neg_t_epoch, auc)
    end_time = time.time()

    return loss_epoch, float(pos_c_epoch) / pos_t_epoch, float(neg_c_epoch) / neg_t_epoch, auc


def saveModel(epoch, net, save_dir, name, args):
    state_dict = net.module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save({
        'epoch': epoch,
        'save_dir': save_dir,
        'state_dict': state_dict,
        'args': args},
        os.path.join(save_dir, name))  # '%03d.ckpt' % epoch))


def train_val(args):
    ################################################
    X_train, Y_train, X_val, Y_val = prepare_trainval_data()

    dataset = FCDataSet(X_train, Y_train, phase='train')
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    dataset = FCDataSet(X_val, Y_val, phase='val')
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    ################################################
    net = FCNet()
    loss = FCLoss()

    if args.gpus is None:
        args.gpus = [0]
    net = net.cuda()
    loss = loss.cuda()
    net = DataParallel(net, device_ids=args.gpus)

    start_epoch = args.start_epoch
    save_dir = args.save_dir

    if args.resume:
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join('results', save_dir)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join('results', args.model + '-' + exp_id)
        else:
            save_dir = os.path.join('results', save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ################################################
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    def get_lr(epoch):
        if epoch <= args.epochs * 0.5:
            lr = args.lr
        elif epoch <= args.epochs * 0.75:
            lr = 0.1 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr

    ################################################
    loss_best = 1.0
    auc_best = 0.5

    ################################################
    viz = Visdom(server="http://localhost", port=11111)
    viz.close(env="main")
    train_opts = dict(legend=["Loss", "TPR", "FPR", "AUC"],
                  markers=True, markersize=5, title='Train', caption='Train', width=1920)
    val_opts = dict(legend=["Loss", "TPR", "FPR", "AUC"],
                  markers=True, markersize=5, title='Val', caption='Val', width=1920)

    NUM_PARAMS = 4
    train_win = viz.line(
        X=np.column_stack([[0]] * NUM_PARAMS),
        Y=np.column_stack([[0]] * NUM_PARAMS),
        opts=train_opts)

    val_win = viz.line(
        X=np.column_stack([[0]] * NUM_PARAMS),
        Y=np.column_stack([[0]] * NUM_PARAMS),
        opts=val_opts)

    ################################################
    for epoch in range(start_epoch, args.epochs + 1):
        _notice(epoch)
        loss_epoch, tpr, fpr, auc = FCTrain(train_loader, net, loss, epoch, optimizer, get_lr)
        viz.line(win=train_win, X=np.column_stack([[epoch], [epoch], [epoch], [epoch]]),
                 Y=np.column_stack([[loss_epoch], [tpr], [fpr], [auc]]), update="append", opts=train_opts)

        loss_epoch, tpr, fpr, auc = FCValidate(val_loader, net, loss)
        viz.line(win=val_win, X=np.column_stack([[epoch], [epoch], [epoch], [epoch]]),
                 Y=np.column_stack([[loss_epoch], [tpr], [fpr], [auc]]), update="append", opts=train_opts)

        # save every save_freq epoch, or the best epoch ever
        if (loss_epoch < loss_best) & (epoch > 20):
            loss_best = loss_epoch
        # if (auc > auc_best) & (epoch > 20):
        #     auc_best = auc
            name = 'model_best.ckpt'
            saveModel(epoch, net, save_dir, name, args)

        if epoch % args.save_freq == 0:
            name = '%03d.ckpt' % epoch
            saveModel(epoch, net, save_dir, name, args)


def test(args):
    if args.resume:
        ##########################################
        # setting up net
        checkpoint = torch.load(args.resume)
        net = FCNet()
        loss = FCLoss()
        if args.gpus is None:
            args.gpus = [0]
        net.load_state_dict(checkpoint['state_dict'])
        net = net.cuda()
        net = DataParallel(net, device_ids=args.gpus)
        net.eval()

        ##########################################
        # setting up data
        X_test, D_name, PhyD, PhySz = prepare_test_data()
        dataset = FCDataSet(X_test, D_name, PhyD, PhySz, phase='test')

        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=False)

        predlist = []
        purelist = []
        # NOTE, numpy returns 1x1 element instead of 1x element
        for i, (x, y_name, dim, vol) in enumerate(data_loader):
            x = Variable(x).cuda()
            casePred, nodPred = net(x)
            ###
            # cols = ["ISBI-PID", "Cohort Label", "Nodule Size at time T1", "Nodule Size at time T2",
            #         "Nodule volume at time T1", "Nodule volume at time T2", "Malignancy Probability",
            #         "Descriptor-Type-Used", "DICOM UIDS at T1", "DICOM UIDs at T2", "Comments"]
            ###
            predlist.append([y_name[0], "TEST", dim[0][0], dim[0][1], vol[0][0], vol[0][1],
                             casePred.data.cpu().numpy()[0],
                             "DeepLearning", "N/A", "N/A", "N/A"])
            purelist.append([y_name[0], casePred.data.cpu().numpy()[0]])
        return predlist, purelist

    else:
        print ("no trained model assigned, quitting......")
        return None


def main():
    ################################################
    parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--save-freq', default='10', type=int, metavar='S',
                        help='save frequency')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_argument('--test', default=0, type=int, metavar='TEST',
                        help='1 do test evaluation, 0 not')
    parser.add_argument('--n_test', default=8, type=int, metavar='N',
                        help='number of gpu for test')
    parser.add_argument('--gpus', nargs='+', type=int, help='use gpu')
    parser.add_argument('--model', '-m', metavar='MODEL', default='base',
                        help='model')

    args = parser.parse_args()

    if args.test == 0:
        train_val(args)
    else:
        predlist, purelist = test(args)

        cols = ["ISBI-PID", "Cohort Label", "Nodule Size at time T1", "Nodule Size at time T2",
                "Nodule volume at time T1", "Nodule volume at time T2", "Malignancy Probability",
                "Descriptor-Type-Used", "DICOM UIDS at T1", "DICOM UIDs at T2",  "Comments"]

        df = pd.DataFrame(predlist, columns=cols)
        df.to_csv("./submission.csv", index=False)

        cols = ["ID", "Malignancy"]
        df = pd.DataFrame(purelist, columns=cols)
        df.to_csv("./isbi_submission.csv", index=False)


if __name__ == "__main__":
    main()






