# Author: Hamza Tazi Bouardi
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from multiclass_utils import get_all_metrics_multiclass


def cart_toolkit(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
):
    cart_model = DecisionTreeClassifier(random_state=0)
    gs_params_cart = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "min_samples_leaf": [5, 10, 15, 20]
    }
    gs_cv_cart = GridSearchCV(cart_model, gs_params_cart, cv=5, n_jobs=-1, scoring="accuracy")
    gs_cv_cart.fit(X_train, y_train)
    results_cart = pd.DataFrame(gs_cv_cart.cv_results_)
    dict_best_params_cart = results_cart[results_cart.rank_test_score == 1]["params"].values[0]
    print("CART \n", dict_best_params_cart)
    cart_model_best = DecisionTreeClassifier(
        criterion=dict_best_params_cart["criterion"],
        max_depth=dict_best_params_cart["max_depth"],
        min_samples_leaf=dict_best_params_cart["min_samples_leaf"],
        random_state=0
    )
    cart_model_best.fit(X_train, y_train)

    # Predict train and test
    y_pred_train_proba_cart = cart_model_best.predict_proba(X_train)
    y_pred_train_cart = cart_model_best.predict(X_train)
    y_pred_cart = cart_model_best.predict(X_test)
    y_pred_proba_cart = cart_model_best.predict_proba(X_test)

    # Generate all useful metrics
    accuracy_train_cart, balanced_accuracy_train_cart, avg_pairwise_auc_train_cart = get_all_metrics_multiclass(
        y_train, y_pred_train_cart, y_pred_train_proba_cart
    )
    accuracy_test_cart, balanced_accuracy_test_cart, avg_pairwise_auc_test_cart = get_all_metrics_multiclass(
        y_test, y_pred_cart, y_pred_proba_cart
    )
    # Scores on train
    print(
        f"CART scores on Train:\nAccuracy={accuracy_train_cart}" +
        f"\t\tBalanced Accuracy={balanced_accuracy_train_cart}" +
        f"\t\tAverage Pairwise AUC={avg_pairwise_auc_train_cart} "
    )
    # Scores on test
    print(
        f"CART scores on Test:\nAccuracy={accuracy_test_cart}" +
        f"\t\tBalanced Accuracy={balanced_accuracy_test_cart}" +
        f"\t\tAverage Pairwise AUC={avg_pairwise_auc_test_cart} "
    )
    return cart_model_best, accuracy_test_cart
