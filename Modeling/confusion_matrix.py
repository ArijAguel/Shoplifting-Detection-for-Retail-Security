import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("/home/Modeling/predictions.csv")

#df["ground_truth"] = df["class_"].apply(lambda x: 0 if x in [1, 2] else 1)

y_true = df["ground_truth"]
y_pred = df["prediction"]

#confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("=== Class Distribution ===")
print(df["ground_truth"].value_counts())
print(df["prediction"].value_counts())

print("\n")
print("=== Confusion Matrix Details ===")
print(f"True Positives: {tp}")
print(f"False Negatives: {fn}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")


print("\nConfusion Matrix: [[TN  FP] [FN TP]]")
print(cm)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n=== Performance Metrics ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")



