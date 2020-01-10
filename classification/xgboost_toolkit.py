# Author: Hamza Tazi Bouardi
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier


def xgboost_toolkit(X_train, y_train, X_test, y_test):
    xgb_model = XGBClassifier(
        random_state=0
    )
    gs_params_xgb = {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.1, 0.05, 0.01, 0.005],
        "max_depth": [2, 3, 4, 5, 6, 7],
        "objective": ['binary:logistic'],
        "n_jobs": [-1]
    }
    gs_cv_obj_xgb = GridSearchCV(xgb_model, gs_params_xgb, cv=5, n_jobs=-1, scoring="roc_auc")
    gs_cv_obj_xgb.fit(X_train, y_train)
    results_xgb = pd.DataFrame(gs_cv_obj_xgb.cv_results_)
    dict_best_params_xgb = results_xgb[results_xgb.rank_test_score == 1]["params"].values[0]
    print("XGBoost \n", dict_best_params_xgb)
    xgb_model_best = XGBClassifier(
        n_estimators=dict_best_params_xgb["n_estimators"],
        learning_rate=dict_best_params_xgb["learning_rate"],
        max_depth=dict_best_params_xgb["max_depth"],
        objective="binary:logistic",
        n_jobs=-1,
        random_state=0
    )
    xgb_model_best.fit(X_train, y_train)
    y_pred_xgb = xgb_model_best.predict(X_test)

    # Predict probabilities instead of only values for AUC
    # Scores on train
    y_pred_train_proba_xgb = xgb_model_best.predict_proba(X_train)[:, 1]
    auc_train_score_xgb = roc_auc_score(y_train, y_pred_train_proba_xgb)
    print(f"XGBoost scores on Train\t AUC={round(auc_train_score_xgb, 3)}")
    # Scores on test
    y_pred_proba_xgb = xgb_model_best.predict_proba(X_test)[:, 1]
    accuracy_test_xgb = accuracy_score(y_test, y_pred_xgb)
    auc_test_score_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    print(f"XGBoost scores on Test " +
          f"set:\t Accuracy={round(accuracy_test_xgb, 2)}\t\t AUC={round(auc_test_score_xgb, 2)}")

    # FPR, TPR and AUC for XGBoost
    fpr_xgb, tpr_xgb, threshold_xgb = roc_curve(y_test, y_pred_proba_xgb)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    return xgb_model_best, roc_auc_xgb, fpr_xgb, tpr_xgb