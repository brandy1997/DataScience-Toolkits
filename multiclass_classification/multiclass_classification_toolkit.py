# Author: Hamza Tazi Bouardi
import click
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from multiclass_cart_toolkit import cart_toolkit
from multiclass_logistic_regression_toolkit import logistic_regression_toolkit
from multiclass_random_forest_toolkit import random_forest_toolkit
from multiclass_xgboost_toolkit import xgboost_toolkit
from multiclass_svm_toolkit import SVM_Toolkit


# We assume that the dataframe we read here is pre-processed,
# with the last column being the labels
df = pd.read_csv("filename.csv")
@click.command()
@click.option(
    '--last_column_is_target',
    prompt='Is the last column of the dataset the target (vector y)? Answer yes/no'
)
@click.option(
    '--random_or_timeseries',
    default="random",
    prompt="Do you want a 'random' train/test split or is this a 'timeseries' dataset?",
    help="Answer with 'random' or 'timeseries'"
)
@click.option(
    '--date_column',
    default="",
    prompt='What is the name of the date column (either str or datetime)?'
)
@click.option(
    '--train_date_limit',
    default="",
    prompt="What is the training set date limit (string format)? This date will be included in train.",
    help="Example: 2020-01-10"
)
def run_multiclass_classification(
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
    threshold_nrows_svm = 5000
    ticks_plot = []
    accuracies_plot = []

    #### CART ####
    cart_model_best, accuracy_cart = cart_toolkit(
        X_train, y_train, X_test, y_test
    )
    ticks_plot += ["CART"]
    accuracies_plot += [accuracy_cart]

    print("###############################################################")
    #### Logistic Regression ####
    logreg_model_best, accuracy_logreg = logistic_regression_toolkit(
        X_train, y_train, X_test, y_test
    )
    ticks_plot += ["Logistic Regression"]
    accuracies_plot += [accuracy_logreg]

    print("###############################################################")
    #### Random Forest ####
    rf_model_best, accuracy_rf = random_forest_toolkit(
        X_train, y_train, X_test, y_test
    )
    ticks_plot += ["Random Forest"]
    accuracies_plot += [accuracy_rf]

    print("###############################################################")
    #### XGBoost ####
    xgb_model_best, accuracy_xgb = xgboost_toolkit(
        X_train, y_train, X_test, y_test
    )
    ticks_plot += ["XGBoost"]
    accuracies_plot += [accuracy_xgb]

    print("###############################################################")
    #### Kernel SVM ####
    if len(X_train) < threshold_nrows_svm:
        SVM_toolkit_obj = SVM_Toolkit(X_train, X_test, y_train, y_test)
        SVM_toolkit_obj.svm_toolkit()
        svm_linear_model_best, accuracy_test_svm_linear = (
            SVM_toolkit_obj.svm_linear_model_best,
            SVM_toolkit_obj.accuracy_test_svm_linear,
        )
        ticks_plot += ["Linear SVM"]
        accuracies_plot += [accuracy_test_svm_linear]

        svm_rbf_model_best, accuracy_test_svm_rbf = (
            SVM_toolkit_obj.svm_rbf_model_best,
            SVM_toolkit_obj.accuracy_test_svm_rbf,
        )
        ticks_plot += ["Gaussian SVM"]
        accuracies_plot += [accuracy_test_svm_rbf]
    else:
        print(f"Number of observations ({len(X_train)}) too big to fit SVMs (will take too much time)")
    print("###############################################################")

    #### Saving plot of Accuracy for all models to compare ####
    models = [i+1 for i in range(len(ticks_plot))]
    plt.figure(figsize=(12, 5))
    plt.title('Comparison of Accuracy for all models', fontsize=16)
    plt.bar(models, accuracies_plot, color='firebrick')
    plt.ylim([0, 1])
    plt.ylabel('Accuracy', fontsize=12, rotation=90)
    plt.xticks(models, ticks_plot)
    plt.savefig(f"{datetime.now().date()}_multiclass_accuracy.png")


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
            assert n_classes > 2, f"This dataset is not fit for multiclass classification as it has {n_classes} classes"
            print(f"Number of classes in dataset: {n_classes}")
        else:
            X_train = df[df[date_column] <= train_date_limit].iloc[:, :-1]
            y_train = df[df[date_column] <= train_date_limit].iloc[:, -1]
            X_test = df[df[date_column] > train_date_limit].iloc[:, :-1]
            y_test = df[df[date_column] > train_date_limit].iloc[:, -1]

    print(f"Train/Test split was performed as {random_or_timeseries}")
    print(f"Percentages of populations:\n{100 * df.iloc[:, -1].value_counts() / len(df)}")
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    run_multiclass_classification()

