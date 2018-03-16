#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-3-13 下午9:19
# @Author  : liuxinglong
# @File    : metastasis_test.py
# @Description: test nodule metastasis
import os
import sys
import pandas as pd
import SimpleITK as sitk
import numpy as np

isbi_csv_file = "./data/labels_isbi_test.csv"
isbi_result = "./isbi_submission.csv"
dsb_result = "./dsb_submission.csv"
submission_result = "./submission.csv"

pair_cnt = 2


def _notice():
    print "*" * 60


def evaluate_metastasis(label_csv):
    # len dataframe return the row count !
    pair_len = len(label_csv) / pair_cnt

    metastasis_id = []
    for idx in range(pair_len):
        base_data = label_csv[idx * pair_cnt: idx * pair_cnt + 1]
        diag_data = label_csv[idx * pair_cnt + 1: idx * pair_cnt + 2]

        base_id = base_data["ID"].item()
        diag_id = diag_data["ID"].item()

        token = base_id.find("_")
        pid = base_id[:token]
        if diag_id.find(pid) < 0:
            print ("pid error in {} -> {}".format(base_id, diag_id))
            continue
        pid = int(pid)

        base_centroid = base_data["Centroid"].item()
        diag_centroid = diag_data["Centroid"].item()

        try:
            base_centroid = np.array(eval(base_centroid))
            diag_centroid = np.array(eval(diag_centroid))
        except Exception, e:
            print ("ERROR, {}".format(e.message))
            continue

        base_voxels = base_data["Voxels"].item()
        diag_voxels = diag_data["Voxels"].item()
        base_radius = pow((3 * float(base_voxels) / (4.0 * np.pi)), 1.0 / 3)
        diag_radius = pow((3 * float(diag_voxels) / (4.0 * np.pi)), 1.0 / 3)

        delta = np.sum(np.power(np.abs(base_centroid - diag_centroid), 2))
        # loose the criteria
        # 2 * pow(base_radius + diag_radius, 2)

        _notice()
        threshold_d = 150
        if delta > threshold_d * threshold_d:
            print ("data {} -> {} out of range, centroid {} -> {}, with radius {} -> {}, total delta {}".format(
                base_id, diag_id, base_centroid, diag_centroid, base_radius, diag_radius, delta))
            metastasis_id.append([pid, base_id, diag_id])
        else:
            print ("data {} -> {} qualified".format(base_id, diag_id))
    return metastasis_id


def main():
    label_csv = pd.read_csv(isbi_csv_file)
    meta_id = evaluate_metastasis(label_csv)

    pd_isbi = pd.read_csv(isbi_result)
    pd_dsb = pd.read_csv(dsb_result)

    pd_sub = pd.read_csv(submission_result)

    for meta_id in meta_id:
        pid = meta_id[0]
        base_id = meta_id[1]
        diag_id = meta_id[2]

        df_isbi = pd_isbi[pd_isbi["ID"] == pid]
        df_sub = pd_sub[pd_sub["ISBI-PID"] == pid]
        df_dsb_base = pd_dsb[pd_dsb["id"] == base_id]
        df_dsb_diag = pd_dsb[pd_dsb["id"] == diag_id]

        pd_sub.at[df_sub.index.item(), "Malignancy Probability"] = df_dsb_diag["cancer"].item()

    # pd_sub["Descriptor-Type-Used"].replace("DeepLearning", "Fixed", inplace=True)
    pd_sub["Descriptor-Type-Used"] = "Fixed"
    pd_sub["DICOM UIDS at T1"] = "N/A"
    pd_sub["DICOM UIDs at T2"] = "N/A"
    pd_sub["Comments"] = "N/A"
    pd_sub.to_csv("./fixed_submission.csv", index=False)

    return


if __name__ == "__main__":
    main()







