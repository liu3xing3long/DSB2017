#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-3-14 下午5:48
# @Author  : liuxinglong
# @File    : fake_selection.py
# @Description: blabla
import os
import numpy as np
import pandas as pd

THRE_H = 0.9
THRE_L = 0.1

src_test= "./fixed_submission_forfakeselection.csv"
src_train = "./labels_isbi.csv"
target = "./labels_isbi_fake.csv"

pd_train = pd.read_csv(src_train)
pd_test = pd.read_csv(src_test)

dat_target = []

pd_test_h = pd_test[pd_test["Malignancy Probability"] >= THRE_H]
pd_test_l = pd_test[pd_test["Malignancy Probability"] <= THRE_L]

basetime = "19990102"
diagtime = "20000102"
# insert original test

for idx, dat_row in pd_train.iterrows():
    dat_target.append([dat_row["ID"], dat_row["Type"], 0, 0, 0, 0, 0])

fake_pos = 0
for idx, dat_row in pd_test_h.iterrows():
    pid = dat_row["ISBI-PID"]
    dat_target.append(["{}_{}".format(pid, basetime), 0, 0, [], 0, 0, 0])
    dat_target.append(["{}_{}".format(pid, diagtime), 1, 0, [], 0, 0, 0])
    fake_pos += 1

fake_neg = 0
for idx, dat_row in pd_test_l.iterrows():
    pid = dat_row["ISBI-PID"]
    dat_target.append(["{}_{}".format(pid, basetime), 0, 0, [], 0, 0, 0])
    dat_target.append(["{}_{}".format(pid, diagtime), 0, 0, [], 0, 0, 0])
    fake_neg += 1

print "adding Neg {}, Pos {}".format(fake_pos, fake_neg)

cols = ["ID", "Type", "Slice", "Centroid", "Voxels", "PhyDiameter", "PhyVolume"]
df = pd.DataFrame(dat_target, columns=cols)
df.to_csv(target, index=False)
























