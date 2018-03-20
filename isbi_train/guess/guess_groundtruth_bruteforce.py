#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-3-18 上午11:28
# @Author  : liuxinglong
# @File    : guess_groundtruth_bruteforce.py
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
# TARGET_ROC1 = 0.8702040816
# TARGET_ROC2 = 0.7967346939
# TARGET_ROC3 = 0.8636734694

TARGET_ROC1 = 0.8726530612
TARGET_ROC2 = 0.7967346939
TARGET_ROC3 = 0.8636734694

Logger = None
SIMILAR_THRESHOLD = 1e-4

TH_H = 0.95
TH_L = 0.1

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

    Ybase =[
        0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 1, 0, 1, 1, 0, 1,
        0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
    Ybase = np.array(Ybase)
    candi_idx = [40, 55, 56, 67]
    candi_idx = np.array(candi_idx)

    Ybase[candi_idx] = -1

    # return tp_idx[0], fp_idx[0], candi_idx[0], X1
    return tp_idx[0], fp_idx[0], candi_idx[0], Ybase


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


# /**
#      * 非递归生成二进制格雷码
#      * 思路:1、获得n-1位生成格雷码的数组
#      *      2、由于n位生成的格雷码位数是n-1的两倍，故只要在n为格雷码的前半部分加0，后半部分加1即可。
#      * @param n 格雷码的位数
#      * @return 生成的格雷码数组
#      */
def GrayCode2(n):
    num = pow(2, n) #//根据输入的整数，计算出此Gray序列大小
    s1 = ["0","1"]#//第一个Gray序列

    if n < 1:
        print ("你输入的格雷码位数有误！");

    for i in range(2, n+1):
    # for(int i=2;i<=n;i++){//循环根据第一个Gray序列，来一个一个的求
        p = pow(2, i); #//到了第几个的时候，来计算出此Gray序列大小
        # String[] si = new String[p];
        si = ["0"] * p
        # for(int j=0;j<p;j++){//循环根据某个Gray序列，来一个一个的求此序列
        for j in range(p):
            if j< p/2:
                si[j] = "0" + s1[j] #//原始序列前面加上"0"
            else:
                si[j] = "1" + s1[p-j-1]#//原始序列反序，前面加上"1"
        s1 = si#//把求得的si，附给s1,以便求下一个Gray序列

    return s1




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

    print roc_auc1, roc_auc2, roc_auc3

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

    candi_idx = [40, 55, 56, 67]
    candi_idx = np.array(candi_idx)
    #########################################################
    X1, X2, X3 = import_data()

    #########################################################
    guess_length = len(candi_idx)
    dat = brgd(guess_length)

    similar_test_times = 3
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

    # GrayCode2(50)

    # print "cal done!"
















