# Author: Hamza Tazi Bouardi
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from binary_utils import get_all_metrics_binary


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # Scaling data (speeds up a lot the fitting)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def logistic_regression_toolkit(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
):
    # Scaling data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Fitting ℓ1 and ℓ2 regularization
    columns = ['params', 'split0_test_score', 'split1_test_score', 'split2_test_score',
               'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score']

    logreg_model = LogisticRegression(random_state=0)
    gs_params_logreg = {
        "penalty": ["l1", "l2"],
        "solver": ["saga"],
        "max_iter": [1000]
    }
    gs_cv_obj_logreg = GridSearchCV(logreg_model, gs_params_logreg, cv=5, n_jobs=-1, scoring="roc_auc")
    gs_cv_obj_logreg.fit(X_train_scaled, y_train)
    results_logreg = pd.DataFrame(gs_cv_obj_logreg.cv_results_)[columns]

    # Fitting elastic net (necessitates other solver)
    logreg_model = LogisticRegression(random_state=0)
    gs_params_logreg_2 = {
        "penalty": ["elasticnet"],
        "solver": ["saga"],
        "l1_ratio": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "max_iter": [1000]
    }
    gs_cv_obj_logreg_2 = GridSearchCV(logreg_model, gs_params_logreg_2, cv=5, n_jobs=-1, scoring="roc_auc")
    gs_cv_obj_logreg_2.fit(X_train_scaled, y_train)
    results_logreg_2 = pd.DataFrame(gs_cv_obj_logreg_2.cv_results_)[columns]
    results_logreg = pd.concat([results_logreg, results_logreg_2], axis=0)
    results_logreg["rank_test_score"] = results_logreg.mean_test_score.rank(method="first", ascending=False)
    dict_best_params_logreg = results_logreg[results_logreg.rank_test_score == 1]["params"].values[0]
    print("Logistic Regression \n", dict_best_params_logreg)

    # The if/else is for either l1/l2 or elasticnet which don't have
    # the same number of parameters in the gridsearch
    if len(dict_best_params_logreg) == len(gs_params_logreg):
        logreg_model_best = LogisticRegression(
            penalty=dict_best_params_logreg["penalty"],
            max_iter=dict_best_params_logreg["max_iter"],
            solver=dict_best_params_logreg["solver"],
            random_state=0
        )
        logreg_model_best.fit(X_train_scaled, y_train)
    else:
        logreg_model_best = LogisticRegression(
            penalty=dict_best_params_logreg["penalty"],
            max_iter=dict_best_params_logreg["max_iter"],
            solver=dict_best_params_logreg["solver"],
            l1_ratio=dict_best_params_logreg["l1_ratio"],
            random_state=0
        )
        logreg_model_best.fit(X_train_scaled, y_train)

    # Predict train and test
    y_pred_train_proba_logreg = logreg_model_best.predict_proba(X_train_scaled)[:, 1]
    y_pred_train_logreg = logreg_model_best.predict(X_train_scaled)
    y_pred_logreg = logreg_model_best.predict(X_test_scaled)
    y_pred_proba_logreg = logreg_model_best.predict_proba(X_test_scaled)[:, 1]

    # Generate all useful metrics
    accuracy_train_logreg, auc_train_logreg, classification_report_train_logreg = get_all_metrics_binary(
        y_train, y_pred_train_logreg, y_pred_train_proba_logreg
    )
    accuracy_test_logreg, auc_test_logreg, classification_report_test_logreg = get_all_metrics_binary(
        y_test, y_pred_logreg, y_pred_proba_logreg
    )

    # Scores on train
    print(f"Logistic Regression scores on Train:\nAccuracy={accuracy_train_logreg} \tAUC={auc_train_logreg}" +
          f"\nClassification Report:\n{classification_report_train_logreg} ")
    # Scores on test
    print(f"Logistic Regression scores on Test:\nAccuracy={accuracy_test_logreg} \tAUC={auc_test_logreg}" +
          f"\nClassification Report:\n{classification_report_test_logreg} ")

    # FPR, TPR and AUC for logreg for visualization
    fpr_logreg, tpr_logreg, threshold_logreg = roc_curve(y_test, y_pred_proba_logreg)
    roc_auc_logreg = auc(fpr_logreg, tpr_logreg)
    return logreg_model_best, roc_auc_logreg, fpr_logreg, tpr_logreg
