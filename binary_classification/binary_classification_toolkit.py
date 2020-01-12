# Author: Hamza Tazi Bouardi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from binary_classification.binary_cart_toolkit import cart_toolkit
from binary_classification.binary_logistic_regression_toolkit import logistic_regression_toolkit
from binary_classification.binary_random_forest_toolkit import random_forest_toolkit
from binary_classification.binary_xgboost_toolkit import xgboost_toolkit
from binary_classification.binary_svm_toolkit import SVM_Toolkit


# We assume that the dataframe we read here is pre-processed,
# with the last column being the labels
df = pd.read_csv("./filename.csv")
X_train, X_test, y_train, y_test = train_test_split(
   df.iloc[:, :-1], df.iloc[:, -1], stratify=df.iloc[:, -1]
)

#### CART ####
cart_model_best, roc_auc_cart, fpr_cart, tpr_cart = cart_toolkit(
    X_train, y_train, X_test, y_test
)

#### Logistic Regression ####
logreg_model_best, roc_auc_logreg, fpr_logreg, tpr_logreg = logistic_regression_toolkit(
    X_train, y_train, X_test, y_test
)

#### Random Forest ####
rf_model_best, roc_auc_rf, fpr_rf, tpr_rf = random_forest_toolkit(
    X_train, y_train, X_test, y_test
)

#### XGBoost ####
xgb_model_best, roc_auc_xgb, fpr_xgb, tpr_xgb = xgboost_toolkit(
    X_train, y_train, X_test, y_test
)

#### Kernel SVM ####
SVM_toolkit_obj = SVM_Toolkit(X_train, X_test, y_train, y_test)
SVM_toolkit_obj.svm_toolkit()
svm_linear_model_best, roc_auc_svm_linear, fpr_svm_linear, tpr_svm_linear = (
    SVM_toolkit_obj.svm_linear_model_best, 
    SVM_toolkit_obj.roc_auc_svm_linear, 
    SVM_toolkit_obj.fpr_svm_linear, 
    SVM_toolkit_obj.tpr_svm_linear
)
svm_rbf_model_best, roc_auc_svm_rbf, fpr_svm_rbf, tpr_svm_rbf = (
    SVM_toolkit_obj.svm_rbf_model_best, 
    SVM_toolkit_obj.roc_auc_svm_rbf, 
    SVM_toolkit_obj.fpr_svm_rbf, 
    SVM_toolkit_obj.tpr_svm_rbf
)


#### Plotting ROC Curve for all models to compare ####
plt.figure(figsize=(12, 7))
plt.title('ROC Curve for all models', fontsize=16)
plt.plot(fpr_cart, tpr_cart, 'b', label='Test AUC CART = %0.3f' % roc_auc_cart, color='g')
plt.plot(fpr_logreg, tpr_logreg, 'b', label='Test AUC LogReg = %0.3f' % roc_auc_logreg, color='orange')
plt.plot(fpr_rf, tpr_rf, 'b', label='Test AUC RF = %0.3f' % roc_auc_rf, color='lime')
plt.plot(fpr_xgb, tpr_xgb, 'b', label='Test AUC XGBoost = %0.3f' % roc_auc_xgb)
plt.plot(fpr_svm_linear, tpr_svm_linear, 'b', label='Test AUC Linear SVM = %0.3f' % roc_auc_svm_linear, color="magenta")
plt.plot(fpr_svm_rbf, tpr_svm_rbf, 'b', label='Test AUC Gaussian SVM = %0.3f' % roc_auc_svm_rbf, color="gray")
plt.plot([0, 1], [0, 1], 'k--', label="Baseline AUC = 0.500")
plt.legend(loc='lower right', fontsize=12)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', fontsize=12)
plt.xlabel('False Positive Rate', fontsize=12)
plt.show()