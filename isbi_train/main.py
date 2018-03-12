import torch
from torch import nn
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from preparedata import prepare_trainval_data


TopK = 5
ConCatCount = 3

class FCNet(nn.Module):
    def __init__(self, topk):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(TopK * 128 * ConCatCount, 64)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
        self.Relu = nn.ReLU()

    def forward(self, centerFeat):
        out = self.dropout(centerFeat)
        out = self.Relu(self.fc1(out))
        out = self.Relu(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        out = out.view(xsize[0],xsize[1])
        base_prob = torch.sigmoid(self.baseline)

        valid_nod_prob = torch.prod(1-out,dim=1)
        dummy_nod_prob = (1-base_prob.expand(out.size()[0]))
        casePred = 1- valid_nod_prob * dummy_nod_prob
        # print "otuput {}".format(out)

        return casePred, out


def FCDataSet(Dataset):
    def __init__(self, X, Y, phase="train"):
        print "working phase {}".format(phase)
        self.X = X
        self.Y = Y
        self.phase = phase

    def __getitem__(idx):
        print "achiving object {}".format(idx)
        return X[idx, :], Y[idx, :]


    


def main():
    X_train, Y_train, X_test, Y_test = prepare_trainval_data()
    print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape



if __name__ == "__main__":
    main()






