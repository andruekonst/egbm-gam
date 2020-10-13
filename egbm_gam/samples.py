import numpy as np


def checkerboard(n_samples=100, scale=2.0):
    """Checkerboard problem.

    @param n_samples Number of samples to generate.
    @param scale Scale of board (how much cells will be on board).
    @return Generated X and y.
    """
    X = np.random.uniform(low=0.0, high=1.0, size=(n_samples, 2))
    y = (np.sin(np.pi * X[:, 0] * scale) > 0) ^ (np.sin(np.pi * X[:, 1] * scale) > 0)
    return X, y


def linear_7dim(n_samples=100, noise_scale=0.05):
    """Simple linear 7-dimensional problem:
    $10 x_1 - 20 x_2 - 2 x_3 + 3 x_4 + Noise$.

    @param n_samples Number of samples to generate.
    @param noise_scale Scale of Gaussian noise.
    @return Generated X and y.
    """
    coef = [10, -20, -2, 3, 0, 0, 0]
    coef = np.array(coef)

    X = np.random.uniform(low=0.0, high=1.0, size=(n_samples, len(coef)))
    y = X.dot(coef)
    y += np.random.normal(scale=noise_scale, size=y.shape)

    return X, y


def nonlinear_7dim(n_samples=100, noise_scale=0.05):
    """Non-linear 7-dimensional problem,
    like simple linear problem, but with quadratic dependence on the last feature:
    $10 x_1 - 20 x_2 - 2 x_3 + 3 x_4 + 100 (x_5 - 0.5) ^ 2 + Noise$.

    @param n_samples Number of samples to generate.
    @param noise_scale Scale of Gaussian noise.
    @return Generated X and y.
    """
    X, y = linear_7dim(n_samples, noise_scale=noise_scale)
    y += 100 * (X[:, -1] - 0.5) ** 2.0
    return X, y


def polynomial_interaction(n_samples=100,
                           n_features=5,
                           n_components=5,
                           degree=2,
                           max_coefficient=100,
                           seed=None):
    """Random polynomial with feature interaction.

    @param n_samples Number of samples to generate.
    @param n_features Number of features.
    @param n_components Number of components in sum.
    @param degree Polynomial degree.
    @param max_coefficient Maximum coefficient value.
    @param seed Random seed.
    @return Generated X and y.
    """
    X = np.random.uniform(low=0.0, high=1.0, size=(n_samples, n_features))
    y = np.zeros((n_samples,))
    rng = np.random.default_rng(seed)

    for i in range(n_components):
        comp_degree = rng.integers(1, degree)
        # coefficient from the normal distribution
        tmp = rng.integers(-max_coefficient, max_coefficient)
        for j in range(comp_degree):
            feature = rng.integers(n_features)
            tmp *= X[:, feature]
        y += tmp

    return X, y


def simple_polynomial(n_samples=100,
                      noise_scale=0.05):
    """Simple polynomial with four dependent features.

    @param n_samples Number of samples to generate.
    @param noise_scale Gaussian noise scale.
    @return Generated X and y.
    """
    n_features = 5
    X = np.random.uniform(low=0.0, high=1.0, size=(n_samples, n_features))

    t = X.T
    r = t[0] ** 2 + t[0] * t[1] - t[2] * t[3] + t[3]

    y = r.T
    y += np.random.normal(0, noise_scale)
    return X, y
