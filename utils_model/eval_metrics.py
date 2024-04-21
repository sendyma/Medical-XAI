from __future__ import print_function
import numpy as np
import os, sys
import random
import time
import glob, cv2

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve


# Generates ROC plot and returns AUC using sklearn
def generate_auc(y_score, y_label):
    fpr, tpr, _ = roc_curve(y_label, y_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(y_label, y_score)
    pr_auc = auc(recall, precision)

    # plt.figure()
    # plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    # plt.plot([0, 1], [0, 1], "k--")
    # plt.xlim([0.0, 1.05])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic curve")
    # plt.show()
    return roc_auc, pr_auc


def compute_roc(probabilities, labels):
    auc, pr_auc = generate_auc(probabilities, labels)
    y_preds = (probabilities > 0.5).astype(np.int)
    se = recall_score(labels, y_preds, pos_label=1)
    sp = recall_score(labels, y_preds, pos_label=0)
    acc = accuracy_score(labels, y_preds)
    return auc, pr_auc, se, sp, acc


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count