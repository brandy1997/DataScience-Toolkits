# Author: Hamza Tazi Bouardi
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression


def logistic_regression_toolkit(X_train, y_train, X_test, y_test):
    # Fitting ℓ1 and ℓ2 regularization
    columns = ['params', 'split0_test_score', 'split1_test_score', 'split2_test_score',
               'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score']

    logreg_model = LogisticRegression(random_state=0)
    gs_params_logreg = {
        "penalty": ["l1", "l2"]
    }
    gs_cv_obj_logreg = GridSearchCV(logreg_model, gs_params_logreg, cv=5, n_jobs=-1, scoring="roc_auc")
    gs_cv_obj_logreg.fit(X_train, y_train)
    results_logreg = pd.DataFrame(gs_cv_obj_logreg.cv_results_)[columns]

    # Fitting elastic net (necessitates other solver)
    logreg_model = LogisticRegression(random_state=0)
    gs_params_logreg_2 = {
        "penalty": ["elasticnet"],
        "solver": ["saga"],
        "l1_ratio": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    }
    gs_cv_obj_logreg_2 = GridSearchCV(logreg_model, gs_params_logreg_2, cv=5, n_jobs=-1, scoring="roc_auc")
    gs_cv_obj_logreg_2.fit(X_train, y_train)
    results_logreg_2 = pd.DataFrame(gs_cv_obj_logreg_2.cv_results_)[columns]
    results_logreg = pd.concat([results_logreg, results_logreg_2], axis=0)
    results_logreg["rank_test_score"] = results_logreg.mean_test_score.rank(method="max", ascending=False)
    dict_best_params_logreg = results_logreg[results_logreg.rank_test_score == 1]["params"].values[0]
    print("Logistic Regression \n", dict_best_params_logreg)

    # The if/else is for either l1/l2 or elasticnet which don't have
    # the same number of parameters in the gridsearch
    if len(dict_best_params_logreg) == 1:
        logreg_model_best = LogisticRegression(penalty=dict_best_params_logreg["penalty"], random_state=0)
        logreg_model_best.fit(X_train, y_train)
    else:
        logreg_model_best = LogisticRegression(
            penalty=dict_best_params_logreg["penalty"],
            solver=dict_best_params_logreg["solver"],
            l1_ratio=dict_best_params_logreg["l1_ratio"],
            random_state=0
        )
        logreg_model_best.fit(X_train, y_train)
    y_pred_logreg = logreg_model_best.predict(X_test)

    # Predict probabilities instead of only values for AUC
    # Scores on train
    y_pred_train_proba_logreg = logreg_model_best.predict_proba(X_train)[:, 1]
    auc_train_score_logreg = roc_auc_score(y_train, y_pred_train_proba_logreg)
    print(f"Logistic Regression scores on Train\t AUC={round(auc_train_score_logreg, 3)}")
    y_pred_proba_logreg = logreg_model_best.predict_proba(X_test)[:, 1]
    # Scores on test
    accuracy_test_logreg = accuracy_score(y_test, y_pred_logreg)
    auc_test_score_logreg = roc_auc_score(y_test, y_pred_proba_logreg)
    print(f"Logistic Regression scores on Test " +
          f"set:\t Accuracy={round(accuracy_test_logreg, 2)}\t\t AUC={round(auc_test_score_logreg, 2)}")

    # FPR, TPR and AUC for Logistic Regression
    fpr_logreg, tpr_logreg, threshold_logreg = roc_curve(y_test, y_pred_proba_logreg)
    roc_auc_logreg = auc(fpr_logreg, tpr_logreg)
    return logreg_model_best, roc_auc_logreg, fpr_logreg, tpr_logreg