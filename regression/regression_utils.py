import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_all_metrics_regression(
        y_train_true: pd.Series,
        y_true: pd.Series,
        y_pred_train: pd.Series,
        y_pred: pd.Series
):
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    mape = round(get_mape(y_true, y_pred), 3)
    mse = round(mean_squared_error(y_true, y_pred), 3)
    isr_squared = round(get_insample_r_squared(y_train_true, y_pred_train), 3)
    osr_squared = round(get_osr_squared(y_train_true, y_true, y_pred), 3)
    return mae, mape, mse, isr_squared, osr_squared


def get_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except ZeroDivisionError:
        mape = np.nan

    return mape


def get_insample_r_squared(y_train_true: pd.Series, y_pred_train: pd.Series) -> float:
    sse_isr = sum([(x - y) ** 2 for x, y in zip(y_train_true, y_pred_train)])
    baseline_avg_train = np.mean(y_train_true)
    sst_isr = sum([(x - baseline_avg_train) ** 2 for x in y_pred_train])
    return 1 - sse_isr / sst_isr


def get_osr_squared(y_train_true: pd.Series, y_true: pd.Series, y_pred: pd.Series) -> float:
    sse_osr = sum([(x - y) ** 2 for x, y in zip(y_true, y_pred)])
    baseline_avg_train = np.mean(y_train_true)
    sst_osr = sum([(x - baseline_avg_train) ** 2 for x in y_pred])
    return 1 - sse_osr / sst_osr
