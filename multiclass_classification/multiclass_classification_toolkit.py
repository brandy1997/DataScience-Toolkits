# Author: Hamza Tazi Bouardi
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from multiclass_classification.multiclass_cart_toolkit import cart_toolkit
from multiclass_classification.multiclass_logistic_regression_toolkit import logistic_regression_toolkit
from multiclass_classification.multiclass_random_forest_toolkit import random_forest_toolkit
from multiclass_classification.multiclass_xgboost_toolkit import xgboost_toolkit
from multiclass_classification.multiclass_svm_toolkit import SVM_Toolkit


# We assume that the dataframe we read here is pre-processed,
# with the last column being the labels
df = pd.read_csv("./filename.csv")
X_train, X_test, y_train, y_test = train_test_split(
   df.iloc[:, :-1], df.iloc[:, -1], stratify=df.iloc[:, -1]
)
assert len(y_train.unique()) > 2, \
    f"This dataset is not fit for multiclass classification as it has {len(y_train.unique())} classes"

#### CART ####
cart_model_best, accuracy_cart = cart_toolkit(
    X_train, y_train, X_test, y_test
)

#### Logistic Regression ####
logreg_model_best, accuracy_logreg = logistic_regression_toolkit(
    X_train, y_train, X_test, y_test
)

#### Random Forest ####
rf_model_best, accuracy_rf = random_forest_toolkit(
    X_train, y_train, X_test, y_test
)

#### XGBoost ####
xgb_model_best, accuracy_xgb = xgboost_toolkit(
    X_train, y_train, X_test, y_test
)

#### Kernel SVM ####
SVM_toolkit_obj = SVM_Toolkit(X_train, X_test, y_train, y_test)
SVM_toolkit_obj.svm_toolkit()
svm_linear_model_best, accuracy_test_svm_linear = (
    SVM_toolkit_obj.svm_linear_model_best, 
    SVM_toolkit_obj.accuracy_test_svm_linear,
)
svm_rbf_model_best, accuracy_test_svm_rbf = (
    SVM_toolkit_obj.svm_rbf_model_best, 
    SVM_toolkit_obj.accuracy_test_svm_rbf,
)


#### Plotting Accuracy for all models to compare ####
ticks = ["CART", "Logistic Regression", "Random Forest", "XGBoost", "Linear SVM", "Gaussian SVM"]
models = [i+1 for i in range(len(ticks))]
accuracies = [
    accuracy_cart, accuracy_logreg, accuracy_rf,
    accuracy_xgb, accuracy_test_svm_linear, accuracy_test_svm_rbf
]
plt.figure(figsize=(12, 5))
plt.title('Comparison of Accuracy for all models', fontsize=16)
plt.bar(models, accuracies, color='firebrick')
plt.ylim([0, 1])
plt.ylabel('Accuracy', fontsize=12, rotation=90)
plt.xticks(models, ticks)
plt.show()
