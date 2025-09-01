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
    precision, recall, thresholds = precision_recall_curve(gt, scores_np)
    auc_pr = auc(recall, precision)

    # 3️⃣ Find best threshold (maximize F1-score)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    # 4️⃣ Convert scores to binary predictions using best threshold
    predictions = (scores_np >= best_threshold).astype(int)

    # 5️⃣ Prepare CSV data
    data = []
    for fname, gt_val, pred_val in zip(filenames, gt, predictions):
        # Correctly split filename: clipName_videoID_pPersonID
        parts = fname.rsplit("_", 2)  # split from right
        clip_id = parts[0]
        video_id = parts[1]
        person_id = parts[2][1:]  # remove leading 'p'
        data.append([clip_id, video_id, person_id, gt_val, pred_val])

    df = pd.DataFrame(data, columns=["clip_id", "video_id", "person_id", "ground_truth", "prediction"])
    df.to_csv(save_csv_path, index=False)

    print(f"CSV saved at {save_csv_path}, Best threshold: {best_threshold:.4f}")
    return auc_roc, auc_pr  # , best_threshold if needed

def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr










