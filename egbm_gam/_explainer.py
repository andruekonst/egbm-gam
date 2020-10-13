from typing import Optional, Union
import numpy as np
import torch

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array

from ._gbms import RGBMRegressor
from sklearn.linear_model import LassoCV


class Explainer(BaseEstimator, RegressorMixin):
    """Generalized Additive Model based on Ensemble of Gradient Boosting Machines.
    """
    def __init__(self,
                 n_epochs: int = 100,
                 weights_lr: float = 1e-2,
                 gbm_lr: float = 1e-2,
                 use_deterministic_trees: bool = False,
                 init_type: Union[str, float] = "target",
                 init_est_type: str = "mean",
                 norm_target: bool = True,
                 optimal_weights: Optional[int] = None,
                 optimal_rate: float = 1.0,
                 optimal_iter: int = 100,
                 optimal_period: Optional[int] = None,
                 optimal_cv_folds: int = 5,
                 pretraining_iter: int = 1,
                 tree_max_depth: Optional[int] = 5,
                 enable_history: bool = False):
        """Construct an explainer.

        @param n_epochs Number of epochs for ensemble training.
        @param weights_lr Learning rates for generalized linear model weights.
        @param gbm_lr Learning rate for GBMs.
        @param use_deterministic_trees Use deterministic trees.
        @param init_type Type of GBMs initialization ("target", "ones" or "zeros");
                         If float number, then use Normal distribution with specified std.
        @param init_est_type Type of a first estimator ("mean" or "linear").
        @param norm_target Normalize target or not.
        @param optimal_weights Recompute linear model weights at each iteration;
                               If is not None, specifies number of iterations before
                               computing optimal weights.
        @param optimal_rate Optimal weights update speed.
        @param optimal_iter Maximum number of iterations for optimal linear model.
        @param optimal_period Period of optimal weights update.
        @param optimal_cv_folds Number of CV folds for LassoCV.
        @param pretraining_iter Number of pretraining iterations.
        @param tree_max_depth Tree maximum depth in GBM.

        @param enable_history Enable history writing during training.
                              History contains weights snapshots for each iteration.
        """
        self.n_epochs = n_epochs
        self.weights_lr = weights_lr
        self.gbm_lr = gbm_lr

        self.use_deterministic_trees = use_deterministic_trees

        self.init_type = init_type
        self.init_est_type = init_est_type
        self.norm_target = norm_target

        # assert(type(optimal_weights) is bool)
        self.optimal_weights = optimal_weights
        self.optimal_rate = optimal_rate
        self.optimal_iter = optimal_iter
        self.optimal_period = optimal_period
        self.optimal_cv_folds = optimal_cv_folds
        self.pretraining_iter = pretraining_iter
        self.tree_max_depth = tree_max_depth

        self.enable_history = enable_history

    def _predict_with_estimators(self, X: np.ndarray, return_torch: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Predict with all estimators.

        @param X Input tensor.
        @param return_torch Return torch tensor if true.
        """
        # TODO: remove code duplication (same as in _predict_last_residuals)
        n_features = X.shape[1]
        preds = [est.predict(X[:, i:i + 1])
                 for i, est in enumerate(self.estimators_[:n_features])]
        preds = np.stack(preds, axis=0)

        if return_torch:
            return torch.tensor(preds, dtype=torch.double, requires_grad=True)
        else:
            return preds

    def _predict_last_residuals(self, X: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        preds = [est.predict_last_residuals(X[:, i:i + 1])
                 for i, est in enumerate(self.estimators_[:n_features])]
        preds = np.stack(preds, axis=0)
        return preds

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'Explainer':
        """Fit ensemble of feature-wise GBMs.

        @param X Training batch inputs.
        @param y Training batch outputs.
        @param sample_weight Sample weights (are not supported for now).
        """
        check_X_y(X, y)

        n_features = X.shape[1]
        gbm_prototype = RGBMRegressor(n_estimators=self.pretraining_iter,
                                      max_depth=self.tree_max_depth,
                                      learning_rate=self.gbm_lr,
                                      init_est_type=self.init_est_type,
                                      use_deterministic_trees=self.use_deterministic_trees)

        # make estimators with the same prototype
        self.estimators_ = [clone(gbm_prototype) for _ in range(n_features)]
        # initialize weights with ones
        self.weights_ = torch.ones(n_features, dtype=torch.double,
                                   requires_grad=True)

        y_norm = 0
        if self.norm_target:
            # normalize target
            self.mean_ = y.mean()
            self.std_ = y.std()
            y_norm = (y - self.mean_) / self.std_
            target = torch.tensor(y_norm).double()
        else:
            target = torch.tensor(y).double()

        # # find center (probably it could be passed as an argument)
        # center = X.mean(axis=0)
        # var = np.mean(X.var(axis=0))
        # # RBF as sample weights
        # sample_weight = np.exp(-((X - center) ** 2.0).sum(axis=1) / (2.0 * var))

        if self.init_type == "target":
            init_target = y if not self.norm_target else y_norm
        elif self.init_type == "ones":
            init_target = np.ones_like(y)
        elif self.init_type == "zeros":
            init_target = np.zeros_like(y)
        elif type(self.init_type) == float:
            init_target = np.random.normal(0.0, self.init_type, size=y.shape)
        else:
            raise ValueError(f"Incorrect init_type: {self.init_type}")

        if self.enable_history:
            self.history_ = []
            self.loss_history_ = []

        # init each gbm
        for i, est in enumerate(self.estimators_):
            est.fit(X[:, i:i + 1], init_target, sample_weight)

        use_opt = (self.optimal_weights is not None)

        outputs = np.zeros_like(X.T)
        # train composition
        for epoch in range(self.n_epochs):
            # compute gbms outputs
            outputs += self._predict_last_residuals(X)

            # check if it is needed to recompute weights
            if use_opt:
                opt_started = (epoch >= self.optimal_weights)
            else:
                opt_started = False

            if self.optimal_period is None:
                opt_period = True
            else:
                opt_period = (epoch % self.optimal_period == 0)

            if use_opt and opt_started and opt_period:
                opt_est = LassoCV(cv=self.optimal_cv_folds)

                opt_est.fit(outputs.T, target.numpy())
                cur_opt_weights = opt_est.coef_.ravel()
                new_weights = torch.tensor(cur_opt_weights, dtype=torch.double,
                                           requires_grad=True)
                # new_intercept = torch.tensor(opt_est.intercept_, dtype=torch.double,
                #                              requires_grad=True)
                self.weights_.data = torch.lerp(self.weights_.data, new_weights,
                                                self.optimal_rate)

                # TODO: check that intercept in regression is close to zero

            cur_outputs = torch.tensor(outputs, dtype=torch.double,
                                       requires_grad=True)

            cumulative_pred = (self.weights_.unsqueeze(1) * cur_outputs).sum(dim=0)

            # calculate loss and gradients
            # MSE loss
            loss = ((target - cumulative_pred) ** 2).mean()
            # loss += self.eta * ((cur_outputs.mean(dim=0) - 1) ** 2).sum().sqrt()
            self.weights_.retain_grad()
            cur_outputs.retain_grad()
            loss.backward()

            # update weights
            self.weights_.data -= self.weights_lr * self.weights_.grad.data

            # update gbms
            for i, est in enumerate(self.estimators_[:n_features]):
                est.append(X[:, i:i + 1], -cur_outputs.grad[i].data.numpy(),
                           sample_weight=sample_weight)

            # clear gradients
            self.weights_.grad.data.zero_()

            # update history
            if self.enable_history:
                self.history_.append(self.weights_.data.numpy().copy())
                self.loss_history_.append(loss.item())

        self.coef_ = self.weights_.data.numpy().copy()

        if self.enable_history:
            self.history_ = np.stack(self.history_, axis=0)

        return self

    def get_corrected_weights(self, X: np.ndarray) -> np.ndarray:
        """Compute corrected weights using provided set.

        @param X Input data.
        @return Corrected weights.
        """
        check_is_fitted(self, attributes=["coef_"])
        # compute standard deviation along each axis
        predictions = self._predict_with_estimators(X, return_torch=False)
        stds = predictions.std(axis=1).ravel()
        return self.coef_ * stds

    def predict_by_feature(self, X: np.ndarray, feature: int) -> np.ndarray:
        """Predict using only GBM corresponding to the feature.

        @param X Input data of shape (n_samples, 1) with only selected feature.
                 If shape is (n_samples,) it will be automatically reshaped.
        @param feature Feature number.
        """
        if X.ndim == 1:
            xs = X.reshape(-1, 1)
        elif X.ndim == 2:
            xs = X
        else:
            raise ValueError(f"Incorrect number of X dimensions: {X.ndim}")
        return self.estimators_[feature].predict(xs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using GAM model.

        @param X Input data.
        @return Array of predictions.
        """
        check_is_fitted(self, attributes=["coef_"])
        check_array(X)

        outputs = self._predict_with_estimators(X)
        cumulative_pred = (self.weights_.unsqueeze(1) * outputs).sum(dim=0)
        predictions = cumulative_pred.data.numpy()
        if self.norm_target:
            return predictions * self.std_ + self.mean_
        else:
            return predictions
