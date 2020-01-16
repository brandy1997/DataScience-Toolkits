# Author: Hamza Tazi Bouardi
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from regression_utils import get_all_metrics_regression


def cart_toolkit(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
):
    cart_model = DecisionTreeRegressor(random_state=0)
    gs_params_cart = {
        "criterion": ["mse", "friedman_mse", "mae"],
        "max_depth": [4, 5, 6, 7, 8, 9, 10],
        "min_samples_leaf": [5, 10, 15, 20],
        "ccp_alpha": np.linspace(0, 0.1, 25)
    }
    gs_cv_cart = GridSearchCV(cart_model, gs_params_cart, cv=5, n_jobs=-1, scoring="neg_mean_squared_error")
    gs_cv_cart.fit(X_train, y_train)
    results_cart = pd.DataFrame(gs_cv_cart.cv_results_)
    dict_best_params_cart = results_cart[results_cart.rank_test_score == 1]["params"].values[0]
    print("CART \n", dict_best_params_cart)
    cart_model_best = DecisionTreeRegressor(
        criterion=dict_best_params_cart["criterion"],
        max_depth=dict_best_params_cart["max_depth"],
        min_samples_leaf=dict_best_params_cart["min_samples_leaf"],
        ccp_alpha=dict_best_params_cart["ccp_alpha"],
        random_state=0
    )
    cart_model_best.fit(X_train, y_train)

    # Predict train and test
    y_pred_train_cart = cart_model_best.predict(X_train)
    y_pred_cart = cart_model_best.predict(X_test)

    # Generate all useful metrics
    mae_cart, mape_cart, mse_cart, isr_squared_cart, osr_squared_cart = get_all_metrics_regression(
        y_train, y_test, y_pred_train_cart, y_pred_cart
    )
    print(
        f"CART scores:\nMAE={mae_cart}" +
        f"\t\tMAPE={mape_cart} %" +
        f"\t\tMSE={mse_cart} " +
        f"\t\tIn-Sample R^2={isr_squared_cart}" +
        f"\t\tOut-of-Sample R^2={osr_squared_cart} "
    )
    return cart_model_best, osr_squared_cart
