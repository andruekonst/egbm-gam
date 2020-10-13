# EGBM-GAM
A straightforward implementation of EGBM-based Generalized Additive Model

## Prerequisites
Make sure, that `pytorch` is installed correctly
(inside python it is called just `torch`).

Package could be installed with `setup.py`:
```
python setup.py install
```

## Usage example
Make a Generalized Additive Model with `Explainer`:
```python
from egbm_gam import Explainer
from egbm_gam.samples import nonlinear_7dim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# generate 7-dimensional regression problem
X, y = nonlinear_7dim(n_samples=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y)

explainer = Explainer(
    n_epochs=2000,
    weights_lr=0.05,
    gbm_lr=0.01,
    init_type='zeros',
    norm_target=True,
    init_est_type='mean',
    optimal_weights=100,
    optimal_rate=0.01,
    optimal_iter=500,
    optimal_period=20,
    pretraining_iter=1,
    tree_max_depth=1,
    enable_history=True
)
# fit global explainer
explainer.fit(X_train, y_train)
```

Trained explanation model could be used to make predictions, just like any other estimator.
It is useful to check that model is fitted to both training and testing datasets:
```python
train_predictions = explainer.predict(X_train)
test_predictions = explainer.predict(X_test)
print('R^2 on Train:', r2_score(y_train, train_predictions))
print('R^2 on Test:', r2_score(y_test, test_predictions))
```

Then we can extract coefficients and corrected weights, which correspond to feature importance:
```python
weights = explainer.coef_
importance = explainer.get_corrected_weights(X_train)
print('Feature importance:', [f'{i:.2}' for i in importance])
```

It is also possible to take a look at dependency on a selected feature:
```python
import numpy as np
from matplotlib import pyplot as plt

feature = 6 # selected feature number (starting from zero)
n_points = 100 # number of points to generate
xs = np.linspace(X_train[:, feature].min(), X_train[:, feature].max(), n_points)
plt.plot(xs, explainer.predict_by_feature(xs, feature))
plt.show()
```
