import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.validation import check_X_y, check_is_fitted
from numpy.linalg import inv


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = X @ self.weights_  # Dot product of X and weights
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution you derived.
        #  Use only numpy functions. Don't forget regularization!

        w_opt = None
        # ====== YOUR CODE: ======
        # Number of features
        n_features = X.shape[1]

        # Create identity matrix of shape (n_features, n_features)
        I = np.eye(n_features)

        # Compute the regularization term, lambda times the identity matrix
        lambda_identity = self.reg_lambda * I

        # Calculate the dot product of X transpose and X
        XTX = np.dot(X.T, X)

        # Calculate the inverse of (X^T * X + lambda * I)
        inv_XTX_lambdaI = inv(XTX + lambda_identity)

        # Calculate the optimal weights: (X^T * X + lambda * I)^(-1) * X^T * y
        w_opt = np.dot(inv_XTX_lambdaI, np.dot(X.T, y))

        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    # TODO: Implement according to the docstring description.
    # ====== YOUR CODE: ======
    # Extract features and target variable
    if feature_names:
        X = df[feature_names]
    else:
        X = df.drop(columns=[target_name])
    y = df[target_name]

    # Fit the model and make predictions
    model.fit(X, y)
    y_pred = model.predict(X)
    # ========================
    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None
        # ====== YOUR CODE: ======
        # Ensure X is a 2D numpy array
        X = check_array(X, ensure_2d=True)

        # Number of samples (rows of X)
        n_samples = X.shape[0]

        # Create a column of ones (bias term)
        ones = np.ones((n_samples, 1))

        # Horizontally stack the column of ones to the original feature matrix
        xb = np.hstack((ones, X))
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        # ========================

    def fit(self, X, y=None):
        # Typically, fitting would determine model parameters. Here, we just initialize our polynomial feature generator.
        self.poly.fit(X)
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        X_poly = self.poly.transform(X)
        # If only polynomial features are needed:
        X_transformed = X_poly
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Extract the correlations of the target feature with all other features
    correlations = correlation_matrix[target_feature]

    # Remove the correlation of the feature with itself
    correlations = correlations.drop(target_feature, errors='ignore')

    # Get the absolute values to find the strongest correlations, regardless of sign
    abs_correlations = correlations.abs()

    # Sort the correlations by absolute value in descending order
    sorted_correlations = abs_correlations.sort_values(ascending=False)

    # Get the top 'n' feature names and their corresponding correlation values
    top_n_features = sorted_correlations.head(n).index.tolist()
    top_n_corr = sorted_correlations.head(n).values.tolist()
    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    # Ensure that the inputs are NumPy arrays for element-wise operations
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    # Calculate the squared differences between predictions and actual values
    squared_diff = (y_pred - y) ** 2
    # Compute the mean of these squared differences
    mse = np.mean(squared_diff)
    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    # Ensure that the inputs are NumPy arrays for element-wise operations
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    # Calculate the sum of squares of residuals
    ss_res = np.sum((y - y_pred) ** 2)

    # Calculate the total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    # Calculate the R^2 score
    r2 = 1 - (ss_res / ss_tot)
    # ========================
    return r2


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range
):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    # Create the pipeline as per your setup
    pipeline = Pipeline([
        ('biastricktransformer', BiasTrickTransformer()),
        ('bostonfeaturestransformer', BostonFeaturesTransformer()),
        ('linearregressor', LinearRegressor())
    ])

    # Set up the parameter grid with corrected names
    param_grid = {
        'bostonfeaturestransformer__degree': degree_range,
        'linearregressor__reg_lambda': lambda_range
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=k_folds,
        verbose=1,
        return_train_score=True
    )

    # Execute the grid search
    grid_search.fit(X, y)

    # Retrieve the best parameters found during the search
    best_params = grid_search.best_params_

    return best_params
