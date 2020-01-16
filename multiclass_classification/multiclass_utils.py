import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def get_all_metrics_multiclass(y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series):
    accuracy = round(accuracy_score(y_true, y_pred), 3)
    balanced_accuracy = round(balanced_accuracy_score(y_true, y_pred), 3)
    avg_pairwise_auc = round(roc_auc_score(y_true=y_true, y_score=y_pred_proba, multi_class='ovo'), 3)
    return accuracy, balanced_accuracy, avg_pairwise_auc
