# DataScience-Toolkits
A bunch of Python scripts of my making in order to apply all kinds of models to an already pre-processed dataset.
Below is a description of all the files in each folder. Note that to use any of these modules, the dataset that is
being called must be already pre-processed, with its last column being the target (y vector). This whole project is
not meant to be a package, but merely an open-source module that can be forked and modified to one's convenience.
- **binary_classification**
    - binary_classification_toolkit: master script for binary classification that contains the script that can be ran. 
    This file calls all the other files in this folder and is meant to be used directly by being called on a terminal
    or by copying/pasting its source code in a Jupyter Notebook cell.
    - binary_cart_toolkit: contains functions used to fit the best CART model for binary classification using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - binary_logistic_regression_toolkit: contains functions used to fit the best CART model for binary classification using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - binary_random_forest_toolkit: contains functions used to fit the best Random Forest model for binary classification using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - binary_svm_toolkit: contains functions used to fit the best SVM (Linear & Gaussian) model for binary classification using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - binary_xgboost_toolkit: contains functions used to fit the best XGBoost model for binary classification using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - binary_utils: utils functions for binary classification (mainly scores and metrics).
    
- **multiclass_classification**
    - multiclass_classification_toolkit: master script for multiclass classification that contains the script that can be ran. 
    This file calls all the other files in this folder and is meant to be used directly by being called on a terminal
    or by copying/pasting its source code in a Jupyter Notebook cell.
    - multiclass_cart_toolkit: contains functions used to fit the best CART model for multiclass classification using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - multiclass_logistic_regression_toolkit: contains functions used to fit the best CART model for multiclass classification using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - multiclass_random_forest_toolkit: contains functions used to fit the best Random Forest model for multiclass classification using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - multiclass_svm_toolkit: contains functions used to fit the best SVM (Linear & Gaussian) model for multiclass classification using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - multiclass_xgboost_toolkit: contains functions used to fit the best XGBoost model for multiclass classification using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - multiclass_utils: utils functions for multiclass classification (mainly scores and metrics).

- **regression**
    - regression_toolkit: master script for regression that contains the script that can be ran. 
    This file calls all the other files in this folder and is meant to be used directly by being called on a terminal
    or by copying/pasting its source code in a Jupyter Notebook cell.
    - regression_linear_regression_and_variants: TO BE DEVELOPED
    - regression_cart_toolkit: contains functions used to fit the best CART model for regression using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - regression_random_forest_toolkit: contains functions used to fit the best Random Forest model for regression using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - regression_xgboost_toolkit: contains functions used to fit the best XGBoost model for regression using 
    gridsearch and cross-validation. _Can be used indepedently from the master script._
    - regression_utils: utils functions for regression (mainly scores and metrics).