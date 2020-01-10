# Author: Hamza Tazi Bouardi
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

def random_forest_toolkit(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(random_state=0)
    gs_params_rf = {
        "n_estimators": [10, 50, 100, 200, 500],
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, 6, 7, 8],
        "min_samples_leaf": [5, 10, 15, 20]
    }
    gs_cv_obj_rf = GridSearchCV(rf_model, gs_params_rf, cv=5, n_jobs=-1, scoring="roc_auc")
    gs_cv_obj_rf.fit(X_train, y_train)
    results_rf = pd.DataFrame(gs_cv_obj_rf.cv_results_)
    dict_best_params_rf = results_rf[results_rf.rank_test_score == 1]["params"].values[0]
    print("Random Forest \n", dict_best_params_rf)
    rf_model_best = RandomForestClassifier(
        n_estimators=dict_best_params_rf["n_estimators"],
        criterion=dict_best_params_rf["criterion"],
        max_depth=dict_best_params_rf["max_depth"],
        min_samples_leaf=dict_best_params_rf["min_samples_leaf"],
        random_state=0
    )
    rf_model_best.fit(X_train, y_train)
    y_pred_rf = rf_model_best.predict(X_test)

    # Predict probabilities instead of only values for AUC
    # Scores on train
    y_pred_train_proba_rf = rf_model_best.predict_proba(X_train)[:, 1]
    auc_train_score_rf = roc_auc_score(y_train, y_pred_train_proba_rf)
    print(f"Random Forest scores on Train\t AUC={round(auc_train_score_rf, 3)}")
    # Scores on test
    y_pred_proba_rf = rf_model_best.predict_proba(X_test)[:, 1]
    accuracy_test_rf = accuracy_score(y_test, y_pred_rf)
    auc_test_score_rf = roc_auc_score(y_test, y_pred_proba_rf)
    print(f"Random Forest scores on Test " +
          f"set:\t Accuracy={round(accuracy_test_rf, 2)}\t\t AUC={round(auc_test_score_rf, 2)}")

    # FPR, TPR and AUC for Random Forest
    fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test, y_pred_proba_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    return rf_model_best, roc_auc_rf, fpr_rf, tpr_rf