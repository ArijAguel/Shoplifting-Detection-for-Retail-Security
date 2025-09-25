import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
import pandas as pd
#from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc
import csv
#from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as auc_func
#from sklearn.metrics import roc_curve
#from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve,auc


def score_auc(scores_np, gt, filenames, save_csv_path="predictions.csv"):
    # 1️⃣ Compute AUC-ROC
    auc_roc = roc_auc_score(gt, scores_np)

    # 2️⃣ Compute Precision-Recall and AUC-PR
    precision, recall, pr_thresholds = precision_recall_curve(gt, scores_np)
    auc_pr = auc(recall, precision)

    # 3️⃣ Compute ROC curve for threshold selection
    fpr, tpr, roc_thresholds = roc_curve(gt, scores_np)

    # Option A: Youden’s J statistic (maximize TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = roc_thresholds[best_idx]

    # Option B (alternative): closest to top-left corner (0,1)
    # distances = np.sqrt(fpr**2 + (1 - tpr)**2)
    # best_idx = np.argmin(distances)
    # best_threshold = roc_thresholds[best_idx]

    # 4️⃣ Convert scores to binary predictions using best threshold
    predictions = (scores_np >= best_threshold).astype(int)

    # 5️⃣ Prepare CSV data
    data = []
    for fname, gt_val, pred_val in zip(filenames, gt, predictions):
        parts = fname.rsplit("_", 2)  # split from right
        clip_id = parts[0]
        video_id = parts[1]
        person_id = parts[2][1:]  # remove leading 'p'
        data.append([clip_id, video_id, person_id, gt_val, pred_val])

    df = pd.DataFrame(data, columns=["clip_id", "video_id", "person_id", "ground_truth", "prediction"])
    df.to_csv(save_csv_path, index=False)

    print(f"CSV saved at {save_csv_path}, Best threshold (ROC-based): {best_threshold:.4f}")
    return auc_roc, auc_pr, best_threshold



def adjusted_score_auc(scores_np, gt, filenames, save_csv_path="adjusted_predictions.csv"):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

    # 1️⃣ Invert scores so higher = more likely abnormal (positive)
    # Actually here, "less negative = abnormal" → just use -scores
    adjusted_scores = -scores_np

    # 2️⃣ Compute AUC-ROC
    auc_roc = roc_auc_score(gt, adjusted_scores)

    # 3️⃣ Compute Precision-Recall and AUC-PR
    precision, recall, pr_thresholds = precision_recall_curve(gt, adjusted_scores)
    auc_pr = auc(recall, precision)

    # 4️⃣ Compute ROC curve for threshold selection
    fpr, tpr, roc_thresholds = roc_curve(gt, adjusted_scores)

    # Youden’s J statistic (maximize TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = roc_thresholds[best_idx]

    # 5️⃣ Convert scores to binary predictions
    # 1 = abnormal, 0 = normal
    predictions = (adjusted_scores >= best_threshold).astype(int)

    # 6️⃣ Prepare CSV data
    data = []
    for fname, gt_val, pred_val in zip(filenames, gt, predictions):
        parts = fname.rsplit("_", 2)  # split from right
        clip_id = parts[0]
        video_id = parts[1]
        person_id = parts[2][1:]  # remove leading 'p'
        data.append([clip_id, video_id, person_id, gt_val, pred_val])

    df = pd.DataFrame(data, columns=["clip_id", "video_id", "person_id", "ground_truth", "prediction"])
    df.to_csv(save_csv_path, index=False)

    print(f"CSV saved at {save_csv_path}, Best threshold (ROC-based): {best_threshold:.4f}")
    return auc_roc, auc_pr, best_threshold


def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr










