# Author: Hamza Tazi Bouardi
import click
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from binary_cart_toolkit import cart_toolkit
from binary_logistic_regression_toolkit import logistic_regression_toolkit
from binary_random_forest_toolkit import random_forest_toolkit
from binary_xgboost_toolkit import xgboost_toolkit
from binary_svm_toolkit import SVM_Toolkit


# We assume that the dataframe we read here is pre-processed,
# with the last column being the labels
df = pd.read_csv("filename.csv")
@click.command()
@click.option('--last_column_is_target',
              prompt='Is the last column of the dataset the target (vector y)? Answer yes/no')
@click.option('--random_or_timeseries', default="random",
              prompt="Do you want a 'random' train/test split or is this a 'timeseries' dataset?",
              help="Answer with 'random' or 'timeseries'")
@click.option('--date_column', default="",
              prompt='What is the name of the date column (either str or datetime)?')
@click.option('--train_date_limit',
              default="",
              prompt="What is the training set date limit (string format)? This date will be included in train.",
              help="Example: 2020-01-10"
              )
def run_binary_classification(
        last_column_is_target,
        random_or_timeseries,
        date_column,
        train_date_limit
):
    X_train, X_test, y_train, y_test = perform_train_test_split(
        last_column_is_target,
        random_or_timeseries,
        date_column,
        train_date_limit
    )
    threshold_nrows_svm = 15000
    #### CART ####
    cart_model_best, roc_auc_cart, fpr_cart, tpr_cart = cart_toolkit(
        X_train, y_train, X_test, y_test
    )
    print("###############################################################")
    #### Logistic Regression ####
    logreg_model_best, roc_auc_logreg, fpr_logreg, tpr_logreg = logistic_regression_toolkit(
        X_train, y_train, X_test, y_test
    )
    print("###############################################################")
    #### Random Forest ####
    rf_model_best, roc_auc_rf, fpr_rf, tpr_rf = random_forest_toolkit(
        X_train, y_train, X_test, y_test
    )
    print("###############################################################")
    #### XGBoost ####
    xgb_model_best, roc_auc_xgb, fpr_xgb, tpr_xgb = xgboost_toolkit(
        X_train, y_train, X_test, y_test
    )
    print("###############################################################")
    #### Kernel SVM ####
    if len(X_train) < threshold_nrows_svm:
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
    else:
        print(f"Number of observations ({len(X_train)}) too big to fit SVMs (will take too much time)")
    print("###############################################################")
    #### Saving ROC Curve for all models to compare ####
    plt.figure(figsize=(12, 7))
    plt.title('ROC Curve for all models', fontsize=16)
    plt.plot(fpr_cart, tpr_cart, 'b', label='Test AUC CART = %0.3f' % roc_auc_cart, color='g')
    plt.plot(fpr_logreg, tpr_logreg, 'b', label='Test AUC LogReg = %0.3f' % roc_auc_logreg, color='orange')
    plt.plot(fpr_rf, tpr_rf, 'b', label='Test AUC RF = %0.3f' % roc_auc_rf, color='lime')
    plt.plot(fpr_xgb, tpr_xgb, 'b', label='Test AUC XGBoost = %0.3f' % roc_auc_xgb)
    if len(X_train) < threshold_nrows_svm:
        plt.plot(fpr_svm_linear, tpr_svm_linear, 'b', label='Test AUC Linear SVM = %0.3f' % roc_auc_svm_linear, color="magenta")
        plt.plot(fpr_svm_rbf, tpr_svm_rbf, 'b', label='Test AUC Gaussian SVM = %0.3f' % roc_auc_svm_rbf, color="gray")
    else:
        pass
    plt.plot([0, 1], [0, 1], 'k--', label="Baseline AUC = 0.500")
    plt.legend(loc='lower right', fontsize=12)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.savefig(f"{datetime.now().date()}_roc_auc.png")


def perform_train_test_split(
        last_column_is_target,
        random_or_timeseries,
        date_column,
        train_date_limit
):
    assert random_or_timeseries in ["random", "timeseries"], "Wrong input for random/timeseries"
    assert last_column_is_target in ["yes", "no"], "Wrong input for last_column_is_target (yes/no)"
    if last_column_is_target == "no":
        raise Exception("Please put last column as target")
    else:
        if random_or_timeseries == "random":
            X_train, X_test, y_train, y_test = train_test_split(
                df.iloc[:, :-1], df.iloc[:, -1], stratify=df.iloc[:, -1]
            )
            n_classes = len(y_train.unique())
            assert n_classes == 2, f"This dataset is not fit for binary classification as it has {n_classes} classes"
        else:
            X_train = df[df[date_column] <= train_date_limit].iloc[:, :-1]
            y_train = df[df[date_column] <= train_date_limit].iloc[:, -1]
            X_test = df[df[date_column] > train_date_limit].iloc[:, :-1]
            y_test = df[df[date_column] > train_date_limit].iloc[:, -1]

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    run_binary_classification()
