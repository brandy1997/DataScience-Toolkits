# Author: Hamza Tazi Bouardi
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from regression_utils import get_all_metrics_regression


def random_forest_toolkit(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
):
    rf_model = RandomForestRegressor(random_state=0)
    gs_params_rf = {
        "n_estimators": [10, 50, 100, 200, 500, 1000],
        "criterion": ["mse", "mae"],
        "max_depth": [2, 3, 4, 5, 6, 7, 8],
        "min_samples_leaf": [5, 10, 15, 20],
        "n_jobs": [-1]
    }
    gs_cv_obj_rf = GridSearchCV(rf_model, gs_params_rf, cv=5, n_jobs=-1, scoring="neg_mean_squared_error")
    gs_cv_obj_rf.fit(X_train, y_train)
    results_rf = pd.DataFrame(gs_cv_obj_rf.cv_results_)
    dict_best_params_rf = results_rf[results_rf.rank_test_score == 1]["params"].values[0]
    print("Random Forest \n", dict_best_params_rf)
    rf_model_best = RandomForestRegressor(
        n_estimators=dict_best_params_rf["n_estimators"],
        criterion=dict_best_params_rf["criterion"],
        max_depth=dict_best_params_rf["max_depth"],
        min_samples_leaf=dict_best_params_rf["min_samples_leaf"],
        random_state=0
    )
    rf_model_best.fit(X_train, y_train)

    # Predict train and test
    y_pred_train_rf = rf_model_best.predict(X_train)
    y_pred_rf = rf_model_best.predict(X_test)

    # Generate all useful metrics
    mae_rf, mape_rf, mse_rf, isr_squared_rf, osr_squared_rf = get_all_metrics_regression(
        y_train, y_test, y_pred_train_rf, y_pred_rf
    )
    print(
        f"Random Forest scores:\nMAE={mae_rf}" +
        f"\t\tMAPE={mape_rf} %" +
        f"\t\tMSE={mse_rf} " +
        f"\t\tIn-Sample R^2={isr_squared_rf}" +
        f"\t\tOut-of-Sample R^2={osr_squared_rf} "
    )
    return rf_model_best, osr_squared_rf
