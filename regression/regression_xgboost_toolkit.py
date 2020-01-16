# Author: Hamza Tazi Bouardi
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from regression_utils import get_all_metrics_regression


def xgboost_toolkit(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
):
    xgb_model = XGBRegressor(
        random_state=0
    )
    gs_params_xgb = {
        "n_estimators": [50, 100, 200, 500, 1000],
        "learning_rate": [0.1, 0.05, 0.01, 0.005, 0.001],
        "max_depth": [2, 3, 4, 5, 6, 7],
        "objective": ['reg:squarederror'],
        "n_jobs": [-1]
    }
    gs_cv_obj_xgb = GridSearchCV(xgb_model, gs_params_xgb, cv=5, n_jobs=-1, scoring="accuracy")
    gs_cv_obj_xgb.fit(X_train, y_train)
    results_xgb = pd.DataFrame(gs_cv_obj_xgb.cv_results_)
    dict_best_params_xgb = results_xgb[results_xgb.rank_test_score == 1]["params"].values[0]
    print("XGBoost \n", dict_best_params_xgb)
    xgb_model_best = XGBRegressor(
        n_estimators=dict_best_params_xgb["n_estimators"],
        learning_rate=dict_best_params_xgb["learning_rate"],
        max_depth=dict_best_params_xgb["max_depth"],
        objective=dict_best_params_xgb["objective"],
        n_jobs=-1,
        random_state=0
    )
    xgb_model_best.fit(X_train, y_train)

    # Predict train and test
    y_pred_train_xgb = xgb_model_best.predict(X_train)
    y_pred_xgb = xgb_model_best.predict(X_test)

    # Generate all useful metrics
    mae_xgb, mape_xgb, mse_xgb, isr_squared_xgb, osr_squared_xgb = get_all_metrics_regression(
        y_train, y_test, y_pred_train_xgb, y_pred_xgb
    )
    print(
        f"XGBoost scores:\nMAE={mae_xgb}" +
        f"\t\tMAPE={mape_xgb} %" +
        f"\t\tMSE={mse_xgb} " +
        f"\t\tIn-Sample R^2={isr_squared_xgb}" +
        f"\t\tOut-of-Sample R^2={osr_squared_xgb} "
    )
    return xgb_model_best, osr_squared_xgb