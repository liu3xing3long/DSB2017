#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-3-12 下午3:03
# @Author  : liuxinglong
# @File    : evaluate_utils.py
# @Description: blabla

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt


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


def calculate_auc(Yval, Ypred):
    # calculating
    fpr, tpr, _ = metrics.roc_curve(Yval, Ypred)
    roc_auc = metrics.auc(fpr, tpr)

    return fpr, tpr, roc_auc