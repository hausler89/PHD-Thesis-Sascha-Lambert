"""Chapter 3: a small neural-network surrogate for the force-strain curve.

The network maps (elapsed time, driving velocity) -> force. It is trained on observed force
trajectories and can replace the theoretical force curve (eq. 3.3) in the generative model,
e.g. for estimating the marginal rejection probability (section 3.3.2). It only predicts the force,
not the unbinding: the bond model stays the Bell-Evans rate.

The network is scikit-learn's ``MLPRegressor`` (one hidden layer of 100 units).
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def features(trajectories, velocities, dt):
    """Stack all data points of all trajectories into (X, y) with X = [time, velocity, strain], y = force.

    The strain ``v * t`` follows from time and velocity; giving it to the network explicitly makes
    learning the curve much easier.
    """
    X = []
    for F, v in zip(trajectories, velocities):
        t = np.arange(len(F)) * dt
        X.append(np.column_stack([t, np.full(len(F), v), v * t]))
    return np.concatenate(X), np.concatenate(trajectories)


def fit_force_surrogate(trajectories, velocities, dt, seed=12345, hidden=100, activation="relu", max_iter=500):
    """Train the surrogate F(t, v) (one hidden layer, Adam, L2 penalty 1e-4); returns a fitted scikit-learn pipeline (``.predict(X)``, ``.score(X, y)``)."""
    X, y = features(trajectories, velocities, dt)
    model = Pipeline([("scaler", StandardScaler()),
                      ("net", MLPRegressor(hidden_layer_sizes=(hidden,), activation=activation,
                                           max_iter=max_iter, random_state=seed))])
    return model.fit(X, y)
