# Author: Hamza Tazi Bouardi
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC


# We assume that the dataframe we read here is pre-processed,
# with the last column being the labels
df = pd.read_csv("./filename.csv")
X_train, X_test, y_train, y_test = train_test_split(
   df.iloc[:, :-1], df.iloc[:, -1], stratify=df.iloc[:, -1]
)

#### CART ####
cart_model = DecisionTreeClassifier(random_state=0)
gs_params_cart = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_samples_leaf": [5, 10, 15, 20]
}
gs_cv_cart = GridSearchCV(cart_model, gs_params_cart, cv=5, n_jobs=-1, scoring="roc_auc")
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
y_pred_cart = cart_model_best.predict(X_test)

# Predict probabilities instead of only values for AUC
y_pred_proba_cart = cart_model_best.predict_proba(X_test)[:, 1]
accuracy_test_cart = accuracy_score(y_test, y_pred_cart)
auc_test_score_cart = roc_auc_score(y_test, y_pred_proba_cart)
# Scores on train
y_pred_train_proba_cart = cart_model_best.predict_proba(X_train)[:, 1]
auc_train_score_cart = roc_auc_score(y_train, y_pred_train_proba_cart)
print(f"CART scores on Train\t AUC={round(auc_train_score_cart,3)}")
# Scores on test
print(f"Decision Tree scores on Test " +
      f"set:\t Accuracy={round(accuracy_test_cart,2)}\t\t AUC={round(auc_test_score_cart,2)}")

# FPR, TPR and AUC for CART
fpr_cart, tpr_cart, threshold_cart = roc_curve(y_test, y_pred_proba_cart)
roc_auc_cart = auc(fpr_cart, tpr_cart)


#### Logistic Regression ####
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
print(f"Logistic Regression scores on Train\t AUC={round(auc_train_score_logreg,3)}")
y_pred_proba_logreg = logreg_model_best.predict_proba(X_test)[:, 1]
# Scores on test
accuracy_test_logreg = accuracy_score(y_test, y_pred_logreg)
auc_test_score_logreg = roc_auc_score(y_test, y_pred_proba_logreg)
print(f"Logistic Regression scores on Test " +
      f"set:\t Accuracy={round(accuracy_test_logreg,2)}\t\t AUC={round(auc_test_score_logreg,2)}")

# FPR, TPR and AUC for Logistic Regression
fpr_logreg, tpr_logreg, threshold_logreg = roc_curve(y_test, y_pred_proba_logreg)
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)


#### Random Forest ####
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
print(f"Random Forest scores on Train\t AUC={round(auc_train_score_rf,3)}")
# Scores on test
y_pred_proba_rf = rf_model_best.predict_proba(X_test)[:, 1]
accuracy_test_rf = accuracy_score(y_test, y_pred_rf)
auc_test_score_rf = roc_auc_score(y_test, y_pred_proba_rf)
print(f"Random Forest scores on Test " +
      f"set:\t Accuracy={round(accuracy_test_rf,2)}\t\t AUC={round(auc_test_score_rf,2)}")

# FPR, TPR and AUC for Random Forest
fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)


#### XGBoost ####
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
print(f"XGBoost scores on Train\t AUC={round(auc_train_score_xgb,3)}")
# Scores on test
y_pred_proba_xgb = xgb_model_best.predict_proba(X_test)[:, 1]
accuracy_test_xgb = accuracy_score(y_test, y_pred_xgb)
auc_test_score_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
print(f"XGBoost scores on Test " +
      f"set:\t Accuracy={round(accuracy_test_xgb,2)}\t\t AUC={round(auc_test_score_xgb,2)}")

# FPR, TPR and AUC for XGBoost
fpr_xgb, tpr_xgb, threshold_xgb = roc_curve(y_test, y_pred_proba_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)


#### Kernel SVM ####
# Scaling data (speeds up a lot the fitting)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Linear Kernel
svm_linear_model = SVC(
    kernel="linear",
    random_state=0
)
gs_params_svm_linear = {
    "C": np.linspace(0, 10, 50),
}
gs_cv_obj_svm_linear = GridSearchCV(svm_linear_model, gs_params_svm_linear, cv=3, n_jobs=-1, scoring="roc_auc")
gs_cv_obj_svm_linear.fit(X_train_scaled, y_train)
results_svm_linear = pd.DataFrame(gs_cv_obj_svm_linear.cv_results_)
dict_best_params_svm_linear = results_svm_linear[results_svm_linear.rank_test_score == 1]["params"].values[0]
print("Linear SVM \n", dict_best_params_svm_linear)
svm_model_best_linear = SVC(
    kernel="linear",
    C=dict_best_params_svm_linear["C"],
    random_state=0,
    probability=True
)
svm_model_best_linear.fit(X_train_scaled, y_train)
y_pred_svm_linear = svm_model_best_linear.predict(X_test_scaled)

# Predict probabilities instead of only values for AUC
# Scores on train
y_pred_train_proba_svm_linear = svm_model_best_linear.predict_proba(X_train_scaled)[:, 1]
auc_train_score_svm_linear = roc_auc_score(y_train, y_pred_train_proba_svm_linear)
print(f"SVM Linear scores on Train\t AUC={round(auc_train_score_svm_linear,3)}")
# Scores on test
y_pred_proba_svm_linear = svm_model_best_linear.predict_proba(X_test_scaled)[:, 1]
accuracy_test_svm_linear = accuracy_score(y_test, y_pred_svm_linear)
auc_test_score_svm_linear = roc_auc_score(y_test, y_pred_proba_svm_linear)
print(f"SVM Linear scores on Test " +
      f"set:\t Accuracy={round(accuracy_test_svm_linear,2)}\t\t AUC={round(auc_test_score_svm_linear,2)}")

# FPR, TPR and AUC for Linear SVM
fpr_svm_linear, tpr_svm_linear, threshold_svm_linear = roc_curve(y_test, y_pred_proba_svm_linear)
roc_auc_svm_linear = auc(fpr_svm_linear, tpr_svm_linear)


## Gaussian Kernel
svm_rbf_model = SVC(
    kernel="rbf",
    random_state=0
)
gs_params_svm_rbf = {
    "C": np.linspace(9, 50, 60),
}
gs_cv_obj_svm_rbf = GridSearchCV(svm_rbf_model, gs_params_svm_rbf, cv=3, n_jobs=-1, scoring="roc_auc")
gs_cv_obj_svm_rbf.fit(X_train_scaled, y_train)
results_svm_rbf = pd.DataFrame(gs_cv_obj_svm_rbf.cv_results_)
dict_best_params_svm_rbf = results_svm_rbf[results_svm_rbf.rank_test_score == 1]["params"].values[0]
print("Gaussian SVM \n", dict_best_params_svm_rbf)
svm_model_best_rbf = SVC(
    kernel="rbf",
    C=dict_best_params_svm_rbf["C"],
    random_state=0,
    probability=True
)
svm_model_best_rbf.fit(X_train_scaled, y_train)
y_pred_svm_rbf = svm_model_best_rbf.predict(X_test_scaled)
# Predict probabilities instead of only values for AUC
# Scores on train
y_pred_train_proba_svm_rbf = svm_model_best_rbf.predict_proba(X_train_scaled)[:, 1]
auc_train_score_svm_rbf = roc_auc_score(y_train, y_pred_train_proba_svm_rbf)
print(f"SVM Gaussian scores on Train\t AUC={round(auc_train_score_svm_rbf,3)}")
# Scores on test
y_pred_proba_svm_rbf = svm_model_best_rbf.predict_proba(X_test_scaled)[:, 1]
accuracy_test_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)
auc_test_score_svm_rbf = roc_auc_score(y_test, y_pred_proba_svm_rbf)
print(f"SVM Gaussian scores on Test " +
      f"set:\t Accuracy={round(accuracy_test_svm_rbf,2)}\t\t AUC={round(auc_test_score_svm_rbf,2)}")
# FPR, TPR and AUC for Gaussian SVM
fpr_svm_rbf, tpr_svm_rbf, threshold_svm_rbf = roc_curve(y_test, y_pred_proba_svm_rbf)
roc_auc_svm_rbf = auc(fpr_svm_rbf, tpr_svm_rbf)


#### Plotting ROC Curve for all models to compare ####
plt.figure(figsize=(12, 7))
plt.title('ROC Curve for all models', fontsize=16)
plt.plot(fpr_cart, tpr_cart, 'b', label='Test AUC CART = %0.3f' % roc_auc_cart, color='g')
plt.plot(fpr_logreg, tpr_logreg, 'b', label='Test AUC LogReg = %0.3f' % roc_auc_logreg, color='orange')
plt.plot(fpr_rf, tpr_rf, 'b', label='Test AUC RF = %0.3f' % roc_auc_rf, color='lime')
plt.plot(fpr_xgb, tpr_xgb, 'b', label='Test AUC XGBoost = %0.3f' % roc_auc_xgb)
plt.plot(fpr_svm_linear, tpr_svm_linear, 'b', label='Test AUC Linear SVM = %0.3f' % roc_auc_svm_linear, color="magenta")
plt.plot(fpr_svm_rbf, tpr_svm_rbf, 'b', label='Test AUC Gaussian SVM = %0.3f' % roc_auc_svm_rbf, color="gray")
plt.plot(fpr_svm, tpr_svm, 'b', label='Test AUC Linear SVM = %0.3f' % roc_auc_svm, color='purple')
plt.plot(fpr_svm, tpr_svm, 'b', label='Test AUC Sigmoid SVM = %0.3f' % roc_auc_svm, color='turquoise')
plt.plot([0, 1], [0, 1],'k--', label="Baseline AUC = 0.500")
plt.legend(loc='lower right', fontsize=12)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', fontsize=12)
plt.xlabel('False Positive Rate', fontsize=12)
plt.show()