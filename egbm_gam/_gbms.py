from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array
# from joblib import Parallel, delayed
# from sklearn.model_selection import train_test_split
#
# from sklearn.metrics import r2_score
# from sklearn.model_selection import cross_val_score
# import scipy.special as sc

from sklearn.dummy import DummyRegressor
# noinspection PyProtectedMember
from sklearn.ensemble import _gb_losses
from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression


class SimpleGBM(BaseEstimator, RegressorMixin):
    """ Simple GBM that uses custom base estimators.
    """
    def __init__(self, init_estimator=None, base_estimator=None,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 loss: str = 'ls'):
        self.init_estimator = init_estimator
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.loss = loss
        self.learning_rate = learning_rate
        
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fit GBM.

        @param X Input data.
        @param y Training target.
        @param sample_weight Sample weights.
        """
        check_X_y(X, y)

        if self.init_estimator is None:
            init_estimator = DummyRegressor()
        else:
            init_estimator = clone(self.init_estimator)
        base_estimator = clone(self.base_estimator)
        loss = _gb_losses.LeastSquaresError(n_classes=1)

        self.gammas_ = []
        self.seq_ = []

        self.seq_.append(init_estimator.fit(X, y, sample_weight))
        self.gammas_.append(1)

        # cumulative predictions
        cum_pred: int = 0
        
        for i in range(self.n_estimators):
            cum_pred += self.gammas_[-1] * self.seq_[-1].predict(X)
            residuals = loss.negative_gradient(y, cum_pred)
            est = clone(base_estimator)
            est.fit(X, residuals, sample_weight=sample_weight)
            self.seq_.append(est)
            self.gammas_.append(self.learning_rate)
            
        return self
    
    def append(self, X: np.ndarray, r: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Append new tree, approximating residuals.
        
        @param X Input data.
        @param r Residuals.
        @param sample_weight Sample weights.
        """
        est = clone(self.base_estimator)
        est.fit(X, r, sample_weight=sample_weight)
        self.seq_.append(est)
        self.gammas_.append(self.learning_rate)
    
    def _predict_k(self, X: np.ndarray, k: int = 1) -> np.ndarray:
        """Predict with first `k` estimators.

        @param X Input data.
        @param k Number of estimators.
        @return Array of predictions.
        """
        check_is_fitted(self, attributes=["seq_"])
        check_array(X)

        n_samples = X.shape[0]
        cum_pred = np.zeros(n_samples)
        for i in range(k):
            cum_pred += self.gammas_[i] * self.seq_[i].predict(X)
        return cum_pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using all estimators.

        @param X Input data.
        @return Array of predictions.
        """
        check_is_fitted(self, attributes=["seq_"])
        check_array(X)
        return self._predict_k(X, k=len(self.seq_))
    
    def predict_last_residuals(self, X: np.ndarray) -> np.ndarray:
        """Predict using only last estimator.

        @param X Input data.
        @return Array of predictions.
        """
        check_is_fitted(self, attributes=["seq_"])
        check_array(X)
        return self.gammas_[-1] * self.seq_[-1].predict(X)


class RGBMRegressor(BaseEstimator, RegressorMixin):
    """GBM with Partially Randomized Decision Trees.
    """
    def __init__(self, n_estimators: int = 100,
                 max_depth: Optional[int] = 1,
                 learning_rate: float = 0.1,
                 init_est_type: str = "mean",
                 use_deterministic_trees: bool = False):
        """Initialize model.
        
        @param n_estimators Number of estimators.
        @param max_depth Tree max depth.
        @param learning_rate Learning rate.
        @param init_est_type Initial estimator ("mean" or "linear").
        @param use_deterministic_trees Use deterministic trees;
                                       if False, use randomized trees.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.init_est_type = init_est_type
        self.use_deterministic_trees = use_deterministic_trees

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Fit GBM.

        @param X Input data.
        @param y Training target.
        @param sample_weight Sample weights.
        """
        # configure decision tree prototype
        if self.use_deterministic_trees:
            splitter = "best"
        else:
            splitter = "random"
        dt = DecisionTreeRegressor(max_depth=self.max_depth,
                                   splitter=splitter)

        # setup initial estimator
        if self.init_est_type == "mean":
            init_est = None 
        elif self.init_est_type == "linear":
            init_est = LinearRegression()
        else:
            raise ValueError(f"Incorrect init_est_type: '{self.init_est_type}'")

        self.gbm_ = SimpleGBM(base_estimator=dt,
                              init_estimator=init_est,
                              n_estimators=self.n_estimators,
                              learning_rate=self.learning_rate)
        self.gbm_.fit(X, y.ravel(), sample_weight)
        return self
    
    def append(self, X: np.ndarray, r: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Append new tree, approximating residuals.
        
        @param X Input data.
        @param r Residuals.
        @param sample_weight Sample weights.
        """
        self.gbm_.append(X, r, sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using all estimators.

        @param X Input data.
        @return Array of predictions.
        """
        check_is_fitted(self, attributes=["gbm_"])
        check_array(X)
        return self.gbm_.predict(X)
    
    def predict_last_residuals(self, X: np.ndarray) -> np.ndarray:
        """Predict using only last estimator.

        @param X Input data.
        @return Array of predictions.
        """
        check_is_fitted(self, attributes=["gbm_"])
        check_array(X)
        return self.gbm_.predict_last_residuals(X)
