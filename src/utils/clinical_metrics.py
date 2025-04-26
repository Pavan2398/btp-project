# ecg_vqa_system/src/utils/clinical_metrics.py
from sklearn.metrics import precision_score, recall_score

def medical_specificity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def medical_sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1)

def clinical_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)