import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def get_all_metrics_binary(y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series):
    accuracy = round(accuracy_score(y_true, y_pred), 3)
    auc = round(roc_auc_score(y_true, y_pred_proba), 3)
    df_classification_report = classification_report(y_true=y_true, y_pred=y_pred)
    return accuracy, auc, df_classification_report
