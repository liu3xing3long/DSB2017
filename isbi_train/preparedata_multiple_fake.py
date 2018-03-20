#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-3-2 下午12:24
# @Author  : liuxinglong
# @File    : skl_learn_test_single_rf.py
# @Description: blabla


# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:17:47 2018

@author: liuxinglong
"""

import pandas as pd
import numpy as np
import random as random
import os

##################################
# dsb2017featfullpath = "/mnt/lustre/liuxinglong/work/DSB2017/features"
dsb2017featfullpath = "../features"
dsb2017featfullpath_candi = "../features_multiple"
data_orig_label_csv = "./data/labels_isbi_multiple_fake.csv"
pair_cnt = 2


##################################
def _notice():
    print "*" * 120


#################################
def process_delta_data(X, Y):
    pair_len = len(X) / pair_cnt
    X_delta = []
    Y_delta = []
    for idx in range(pair_len):
        X_delta.append(X[idx * pair_cnt + 1, :] - X[idx * pair_cnt, :])
        Y_delta.append(Y[idx * pair_cnt + 1] - Y[idx * pair_cnt])

    X_delta = np.array(X_delta)
    Y_delta = np.array(Y_delta)

    return X_delta, Y_delta


def import_data():
    X = []
    Y = []
    orig_labels = pd.read_csv(data_orig_label_csv)

    for idx, item in orig_labels.iterrows():
        item_id = str(item["ID"])

        data_path = os.path.join(dsb2017featfullpath, "{}_feat.npy".format(item_id))
        if os.path.isfile(data_path):
            dat = np.load(data_path)
        else:
            data_path = os.path.join(dsb2017featfullpath_candi, "{}_feat.npy".format(item_id))
            dat = np.load(data_path)

        X.append(dat)
        Y.append(item["Type"])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def split_data(X, Y, X_delta, Y_delta, valcount=3):
    delta_Positives = np.where(Y_delta == 1)[0]
    delta_Negatives = np.where(Y_delta == 0)[0]
    ## final svm
    DELTA_VAL_COUNT = valcount
    delta_data_idx = range(len(Y_delta))
    delta_rand_val1 = random.sample(delta_Positives, DELTA_VAL_COUNT)
    delta_rand_val2 = random.sample(delta_Negatives, DELTA_VAL_COUNT)
    delta_rand_val = np.concatenate((delta_rand_val1, delta_rand_val2))
    delta_rand_train = set(delta_data_idx).difference(set(delta_rand_val))

    delta_rand_val = np.array(list(delta_rand_val))
    delta_rand_train = np.array(list(delta_rand_train))

    # Xt_delta_train = X_delta[delta_rand_train, :]
    # Xt_delta_val = X_delta[delta_rand_val, :]
    # Yt_delta_train = Y_delta[delta_rand_train]
    # Yt_delta_val = Y_delta[delta_rand_val]

    X_final_train = []
    Y_final_train = []

    X_final_val = []
    Y_final_val = []

    print X.shape, Y.shape
    print X_delta.shape, Y_delta.shape
    print ("Neg {}, Pos {}".format(len(delta_Negatives), len(delta_Positives)))

    for idx_train in delta_rand_train:
        X_final_train.append(np.concatenate((X[idx_train * pair_cnt, :, :], X[idx_train * pair_cnt + 1, :, :], X_delta[idx_train, :, :]), 1))
        # X_final_train.append([X[idx_train * pair_cnt, :, :], X[idx_train * pair_cnt + 1, :, :], X_delta[idx_train, :, :]])
        Y_final_train.append(Y_delta[idx_train])

    for idx_val in delta_rand_val:
        X_final_val.append(np.concatenate((X[idx_val * pair_cnt, :, :], X[idx_val * pair_cnt + 1, :, :], X_delta[idx_val, :, :]), 1))
        # X_final_val.append([X[idx_val * pair_cnt, :, :], X[idx_val * pair_cnt + 1, :, :], X_delta[idx_val, :, :]])
        Y_final_val.append(Y_delta[idx_val])

    # Loss needs float tensor for 'target' and need double tensor for 'data'!
    X_final_train = np.array(X_final_train)
    Y_final_train = np.array(Y_final_train, dtype=np.float32)
    X_final_val = np.array(X_final_val)
    Y_final_val = np.array(Y_final_val, dtype=np.float32)

    return X_final_train, Y_final_train, X_final_val, Y_final_val


def prepare_trainval_data():
    # import data
    X, Y = import_data()
    # process delta data
    X_delta, Y_delta = process_delta_data(X, Y)
    # split data and combine features
    X_train, Y_train, X_val, Y_val = split_data(X, Y, X_delta, Y_delta, 1)

    # debug
    # print X_train.shape, Y_train.shape, X_val.shape, Y_val.shape

    return X_train, Y_train, X_val, Y_val


def main():
    prepare_trainval_data()


if __name__ == "__main__":
    main()
















