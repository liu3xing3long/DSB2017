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
data_orig_label_csv = "./data/labels_isbi_test.csv"
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

        # Y_delta.append(Y[idx * pair_cnt + 1] - Y[idx * pair_cnt])
        imagename = Y[idx * pair_cnt]
        token = imagename.find("_")
        imagepid = imagename[:token]
        imagetime = imagename[token + 1:]
        imagename2 = Y[idx * pair_cnt + 1]
        if imagename2.find(imagepid) < 0:
            print ("error occured when pairing two delta images, names {}/{}".format(imagename, imagename2))
        else:
            Y_delta.append(imagepid)

    X_delta = np.array(X_delta)
    Y_delta = np.array(Y_delta)

    return X_delta, Y_delta


#################################
def split_data(X, Y, X_delta, Y_delta):
    idx_test_ranges = range(len(Y_delta))

    X_final_test = []
    Y_final_test = []
    for idx_train in idx_test_ranges:
        X_final_test.append(np.concatenate((X[idx_train * pair_cnt, :, :], X[idx_train * pair_cnt + 1, :, :], X_delta[idx_train, :, :]), 1))
        Y_final_test.append(Y_delta[idx_train])
    # Loss needs float tensor for 'target' and need double tensor for 'data'!
    X_final_test = np.array(X_final_test)
    Y_final_test = np.array(Y_final_test)
    return X_final_test, Y_final_test


#################################
def import_data():
    X = []
    Y = []
    orig_labels = pd.read_csv(data_orig_label_csv)

    for idx, item in orig_labels.iterrows():
        item_id = str(item["ID"])
        dat = np.load(os.path.join(dsb2017featfullpath, "{}_feat.npy".format(item_id)))
        X.append(dat)
        # Y.append(item["Type"])
        Y.append(item_id)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def prepare_test_data():
    # import data
    X, Y = import_data()
    # process delta data
    X_delta, Y_delta = process_delta_data(X, Y)
    # in testing, Y_delta contains the LabelName !!
    X_final, Y_final = split_data(X, Y, X_delta, Y_delta)
    # debug
    print X_delta.shape, Y_delta.shape
    return X_final, Y_final


def main():
    X, Y = import_data()
    print "X shape {}, Y shape {}".format(X.shape, Y.shape)


if __name__ == "__main__":
    main()
















