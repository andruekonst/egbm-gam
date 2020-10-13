from typing import Union
import numpy as np


def gen_training_sample(model, n_features: int,
                        n_points: int = 100,
                        mean: Union[float, np.ndarray] = 0.0,
                        var: Union[float, np.ndarray] = 1.0,
                        mode: str = 'uniform',
                        predict_proba: bool = False):
    """Generate training sample for the explanation task.

    @param model Model to be explained.
    @param n_features Number of input features.
    @param n_points Number of points to generate.
    @param mean Point at which explanation is needed.
    @param var Variance or width of explanation field.
    @param mode Mode ('normal' or 'uniform').
    @param predict_proba Predict probabilities (for classification models).

    @return Tuple of generated points and predictions at them.
    """
    if mode == 'normal':
        points = np.random.normal(mean, var, size=(n_points, n_features))
    elif mode == 'uniform':
        low = mean - var / 2.0
        high = mean + var / 2.0
        points = np.random.uniform(low, high, size=(n_points, n_features))
    else:
        raise ValueError(f"Incorrect mode: '{mode}'")

    if predict_proba:
        predictions = model.predict_proba(points)[:, 1]
    else:
        predictions = model.predict(points)
    return points, predictions
