# Author: Hamza Tazi Bouardi
import click
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from regression_cart_toolkit import cart_toolkit
from regression_random_forest_toolkit import random_forest_toolkit
from regression_xgboost_toolkit import xgboost_toolkit


# We assume that the dataframe we read here is pre-processed,
# with the last column being the labels
df = pd.read_csv("../../../1. Fall Semester/15.095 Machine Learning under a modern Optimization Lens/4. Final Project/data/2. Concrete/concrete_complete.csv")
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
def run_regression(
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
    ticks_plot = []
    list_osr_squared_plot = []

    #### CART ####
    cart_model_best, osr_squared_cart = cart_toolkit(
        X_train, y_train, X_test, y_test
    )
    ticks_plot += ["CART"]
    list_osr_squared_plot += [osr_squared_cart]

    print("###############################################################")
    #### Random Forest ####
    rf_model_best, osr_squared_rf = random_forest_toolkit(
        X_train, y_train, X_test, y_test
    )
    ticks_plot += ["Random Forest"]
    list_osr_squared_plot += [osr_squared_rf]

    print("###############################################################")
    #### XGBoost ####
    xgb_model_best, osr_squared_xgb = xgboost_toolkit(
        X_train, y_train, X_test, y_test
    )
    ticks_plot += ["XGBoost"]
    list_osr_squared_plot += [osr_squared_xgb]

    #### Saving plot of Accuracy for all models to compare ####
    models = [i+1 for i in range(len(ticks_plot))]
    plt.figure(figsize=(12, 8))
    plt.title('Comparison of OSR^2 for all models', fontsize=16)
    plt.bar(models, list_osr_squared_plot, color='firebrick')
    plt.ylim([-1, 1])
    plt.ylabel('OSR^2', fontsize=12, rotation=90)
    plt.xticks(models, ticks_plot)
    plt.savefig(f"{datetime.now().date()}_regression_osr_squared.png")


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
                df.iloc[:, :-1], df.iloc[:, -1]
            )
        else:
            X_train = df[df[date_column] <= train_date_limit].iloc[:, :-1]
            y_train = df[df[date_column] <= train_date_limit].iloc[:, -1]
            X_test = df[df[date_column] > train_date_limit].iloc[:, :-1]
            y_test = df[df[date_column] > train_date_limit].iloc[:, -1]

    print(f"Train/Test split was performed as {random_or_timeseries}")
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    run_regression()

