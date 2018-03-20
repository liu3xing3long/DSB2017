#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-3-14 下午2:44
# @Author  : liuxinglong
# @File    : guess_groundtruth.py
# @Description: blabla

import os
import numpy as np
import time
import os
import pandas as pd
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
import logging
# from tqdm import tqdm
import matplotlib.pyplot as plt

submission_groundtruth = "./prediction_GT_only.csv"
submission_file1 = "./prediction_test_yizhong.csv"


def plot_auc(fpr, tpr, roc_auc, figure_idx=1):
    # drawing
    plt.figure(num=figure_idx)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def process_auc(X1, Y):
    fpr, tpr, _ = metrics.roc_curve(Y, X1)
    roc_auc1 = metrics.auc(fpr, tpr)
    print roc_auc1
    plot_auc(fpr, tpr, roc_auc1)


def import_data():
    ################################################
    # data 1
    X1 = []
    data = pd.read_csv(submission_file1)
    for idx, data_piece in data.iterrows():
        X1.append(data_piece["cancer"])
    X1 = np.array(X1)


    ################################################
    Y = []
    data = pd.read_csv(submission_groundtruth)
    for idx, data_piece in data.iterrows():
        Y.append(data_piece["type"])
    Y = np.array(Y)

    Y[Y == -1] = 0

    return X1, Y


def main():

    #########################################################
    X1, Ybase = import_data()

    process_auc(X1, Ybase)


if __name__ == '__main__':
    main()
    # test_data()

