import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

#from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc
import csv
#from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as auc_func
#from sklearn.metrics import roc_curve
#from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc


def score_auc(scores_np, gt):
    auc_roc= roc_auc_score(gt, scores_np)   #Area Under the ROC Curve
    #precision, recall, _ = precision_recall_curve(gt, scores_np)
    #auc_pr = auc_func(recall, precision)
    # auc_pr = average_precision_score(gt, scores_np)
    # fpr, tpr, thresholds = roc_curve(gt, scores_np)  # Get FPR, TPR, thresholds from ROC curve
    # fnr = 1 - tpr  # False Negative Rate = 1 - True Positive Rate

    # # Calculate EER where FPR equals FNR
    # eer_index = np.nanargmin(np.abs(fnr - fpr))  # Find index where FPR = FNR
    # eer_threshold = thresholds[eer_index]  # Threshold at which EER occurs
    # eer = fpr[eer_index]  # Equal Error Rate (where FPR = FNR)
    # return auc, auc_pr, eer, eer_threshold
    precision, recall, thresholds = precision_recall_curve(gt, scores_np)
    auc_precision_recall = auc(recall, precision)
    fpr, tpr, threshold = roc_curve(gt, scores_np, pos_label=1)
    fnr = 1 - tpr
    #eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    #eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return auc_roc, auc_precision_recall#, eer, eer_threshold

def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr










