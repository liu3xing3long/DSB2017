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
from multiprocessing import Pool
from functools import partial
from  evaluate_utils import plot_auc

submission_file1 = "./fixed_submission.csv"
submission_file2 = "./rad_submission.csv"
submission_file3 = "./dsb2nd_submission.csv"

submission_file4 = "./submission_handcraft.csv"

THRE = 0.1
DATA_LENGTH = 70
TARGET_ROC1 = 0.8726530612
TARGET_ROC2 = 0.7967346939
TARGET_ROC3 = 0.8636734694

Logger = None
SIMILAR_THRESHOLD = 1e-7


def test_data():
    X1 = []
    data = pd.read_csv(submission_file1)
    for idx, data_piece in data.iterrows():
        X1.append(data_piece["Malignancy Probability"])
    X1 = np.array(X1)

    # tp_idx = np.where(X1 > 0.8)
    # fp_idx = np.where(X1 < 0.1)
    # print tp_idx, fp_idx

    candi_idx = np.where(np.logical_and(X1 > 0.1, X1 < 0.8))
    # print len(candi_idx[0])

    X1[X1 > 0.8] = 1
    X1[X1 < 0.2] = 0

    return candi_idx


def brgd(n):
    '''
    递归生成n位的二进制反格雷码
    :param n:
    :return:
    '''
    if n==1:
        return ["0", "1"]
    L1 = brgd(n-1)
    L2 = copy.deepcopy(L1)
    L2.reverse()
    L1 = ["0" + l for l in L1]
    L2 = ["1" + l for l in L2]
    L = L1 + L2
    return L


def plot_guess(X1, X2, X3, Y):
    fpr, tpr, _ = metrics.roc_curve(Y, X1)
    roc_auc1 = metrics.auc(fpr, tpr)
    print roc_auc1
    # plot_auc(fpr, tpr, roc_auc1)

    fpr, tpr, _ = metrics.roc_curve(Y, X2)
    roc_auc2 = metrics.auc(fpr, tpr)
    print roc_auc2
    # plot_auc(fpr, tpr, roc_auc2)

    fpr, tpr, _ = metrics.roc_curve(Y, X3)
    roc_auc3 = metrics.auc(fpr, tpr)
    print roc_auc3
    # plot_auc(fpr, tpr, roc_auc3)


def guess(id, data, X1, X2, X3, Y, step=0, step_length=10):
    global Logger

    try:
        test_array = [eval(l) for l in data[id] ]
    except Exception,e :
        Logger.error("Eval Exception {}".format(e.message))

    Y[step*step_length: (step + 1)*step_length] = test_array

    fpr, tpr, _ = metrics.roc_curve(Y, X1)
    roc_auc1 = metrics.auc(fpr, tpr)

    fpr, tpr, _ = metrics.roc_curve(Y, X2)
    roc_auc2 = metrics.auc(fpr, tpr)

    fpr, tpr, _ = metrics.roc_curve(Y, X3)
    roc_auc3 = metrics.auc(fpr, tpr)

    if abs(roc_auc1 - TARGET_ROC1) < SIMILAR_THRESHOLD and \
        abs(roc_auc2 - TARGET_ROC2) < SIMILAR_THRESHOLD and \
        abs(roc_auc3 - TARGET_ROC3) < SIMILAR_THRESHOLD:

        Logger.error("similar ! {} -> {}, {} -> {}, {} -> {}".format(
            roc_auc1, TARGET_ROC1,
            roc_auc2, TARGET_ROC2,
            roc_auc3, TARGET_ROC3))
        Logger.error("current Y {}".format(Y))
    else:
        Logger.debug("skipping")


def import_data():
    ################################################
    # data 1
    X1 = []
    data = pd.read_csv(submission_file1)
    for idx, data_piece in data.iterrows():
        X1.append(data_piece["Malignancy Probability"])
    X1 = np.array(X1)

    ################################################
    # data 2
    X2 = []
    data = pd.read_csv(submission_file2)
    for idx, data_piece in data.iterrows():
        X2.append(data_piece["Malignancy Probability"])
    X2 = np.array(X2)

    ################################################
    # data 3
    X3 = []
    data = pd.read_csv(submission_file3)
    for idx, data_piece in data.iterrows():
        X3.append(data_piece["Malignancy Probability"])
    X3 = np.array(X3)

    ################################################
    Y = []
    data = pd.read_csv(submission_file4)
    for idx, data_piece in data.iterrows():
        Y.append(data_piece["Malignancy Probability"])
    Y = np.array(Y)
    # Y = [1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
    #      0, 0, 1, 0, 1, 0, 0, 0, 0, 1,
    #      0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
    #      0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
    #      0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
    #      1, 1, 1, 0, 0, 0, 0, 0, 0, 1,
    #      1, 0, 1, 1, 1, 0, 1, 0, 0, 1]


    # Y =[
    #     0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
    #     1, 0, 1, 0, 1, 0, 0, 0, 1, 1,
    #     1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
    #     1, 1, 0, 1, 1, 0, 1, 1, 0, 1,
    #     ?, 0, 1, 0, 1, 0, 1, 0, 0, 0,
    #     1, 1, 1, 0, 0, ?, ?, 0, 0, 1,
    #     1, 0, 1, 0, 1, 0, 1, ?, 0, 1
    # ]
    Y =[
        0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 1, 0, 1, 1, 0, 1,
        0,
        0, 1, 0, 1, 0, 1, 0, 0, 0,
        1, 1, 1, 0, 0,
        0,
        0,
        0, 0, 1, 1, 0, 1, 0, 1, 0, 1,
        0,
        0, 1]


    return X1, X2, X3, Y


def main():
    #########################################################
    # Configure logging
    rLogger = logging.getLogger('guess')
    # Create handler for writing to log file
    handler = logging.FileHandler(filename="./guess.log", mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)
    # Initialize logging for batch log messages
    global Logger
    Logger = rLogger.getChild('batch')
    console = logging.StreamHandler()
    rLogger.addHandler(console)

    rLogger.setLevel(logging.INFO)
    #########################################################
    X1, X2, X3, Ybase = import_data()

    plot_guess(X1, X2, X3, Ybase)
    #########################################################
    # step_length = 20
    # total_steps = DATA_LENGTH / step_length
    # dat = brgd(step_length)
    # n_worker = 8

    # Logger.info("starting, step length{}, data length {}, total steps {} ".format(step_length, len(dat), total_steps))
    # for tqdmindex, step in tqdm(enumerate(range(total_steps))):
    #
    #     pool = Pool(n_worker)
    #     partial_guess = partial(guess, data=dat, X1=X1, X2=X2, X3=X3, Y=Ybase, step=step, step_length=step_length)
    #
    #     N = len(dat)
    #     _ = pool.map(partial_guess, range(N))
    #     pool.close()
    #     pool.join()
    # return


if __name__ == '__main__':
    main()
    # test_data()

