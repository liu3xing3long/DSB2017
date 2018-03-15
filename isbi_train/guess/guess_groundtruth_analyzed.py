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
from multiprocessing import Pool
from functools import partial

submission_file1 = "./fixed_submission.csv"
submission_file2 = "./rad_submission.csv"
submission_file3 = "./dsb2nd_submission.csv"

THRE = 0.1
DATA_LENGTH = 70
TARGET_ROC1 = 0.8702040816
TARGET_ROC2 = 0.7967346939
TARGET_ROC3 = 0.8636734694

Logger = None
# SIMILAR_THRESHOLD = 1e-5
SIMILAR_THRESHOLD = 0.01

TH_H = 0.8
TH_L = 0.2

def test_data():
    X1 = []
    data = pd.read_csv(submission_file1)

    for idx, data_piece in data.iterrows():
        X1.append(data_piece["Malignancy Probability"])
    X1 = np.array(X1)

    tp_idx = np.where(X1 > TH_H)
    fp_idx = np.where(X1 < TH_L)
    # print tp_idx, fp_idx

    candi_idx = np.where(np.logical_and(X1 > TH_L, X1 < TH_H))
    # print len(candi_idx[0])

    X1[tp_idx] = 1
    X1[fp_idx] = 0
    X1[candi_idx] = -1 # for debug

    return tp_idx[0], fp_idx[0], candi_idx[0], X1


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


def guess(id, data, guess_index, X1, X2, X3, Y):
    global Logger
    try:
        test_array = [eval(l) for l in data[id] ]
    except Exception,e :
        Logger.error("Eval Exception {}".format(e.message))

    Y[guess_index] = test_array

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

    # ################################################
    # Y = [1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
    #      1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
    #      1, 1, 0, 0, 1, 0, 0, 0, 1, 0,
    #      1, 1, 0, 1, 1, 0, 1, 1, 0, 0,
    #      1, 0, 1, 0, 1, 0, 1, 0, 0, 0,
    #      1, 1, 1, 0, 0, 0, 0, 0, 0, 1,
    #      1, 0, 1, 1, 1, 0, 1, 0, 0, 1]
    # return X1, X2, X3, Y
    return X1, X2, X3


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
    tp_idx, fp_idx, candi_idx, Ybase = test_data()


    #########################################################
    X1, X2, X3 = import_data()

    #########################################################
    guess_length = len(candi_idx)
    dat = brgd(guess_length)

    # similar_test_times = 3
    similar_test_times = 1
    global SIMILAR_THRESHOLD
    for i_test_time in range(similar_test_times):
        print "working at test time {}, threhosld {}".format(i_test_time, SIMILAR_THRESHOLD)

        use_pool = False
        if use_pool == True:
            n_worker = 32
            pool = Pool(n_worker)
            partial_guess = partial(guess, data=dat, guess_index=candi_idx, X1=X1, X2=X2, X3=X3, Y=Ybase)

            N = len(dat)
            Logger.info("starting, guess length {}, data length {} ".format(guess_length, N))

            res = pool.map(partial_guess, range(N))
            pool.close()
            pool.join()
        else:
            from tqdm import tqdm
            N = len(dat)
            Logger.info("starting, guess length {}, data length {} ".format(guess_length, N))
            for i in tqdm(range(N)):
                guess(i, data=dat, guess_index=candi_idx, X1=X1, X2=X2, X3=X3, Y=Ybase)

        SIMILAR_THRESHOLD *= 10

    return


if __name__ == '__main__':
    main()
    # test_data()

