from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import linear_model


def bandit_sl(dataset: pd.DataFrame) -> np.ndarray[int]:
    dosage_buckets = []
    X, y = _compute_features_and_targets(dataset)
    model = _initialize_linear_model(X, y)
    for i in range(len(X)):
        y_pred = model.predict(np.expand_dims(X[i], axis=0))[0]
        if y_pred < 21:
            dosage_buckets.append(0)
        elif y_pred <= 49:
            dosage_buckets.append(1)
        else:
            dosage_buckets.append(2)

        model = model.fit(X[: i + 1], y[: i + 1])
    return np.array(dosage_buckets)


def _compute_features_and_targets(
    dataset: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    features_df = dataset.drop("Therapeutic Dose of Warfarin", axis=1)
    features_np = np.hstack([features_df.to_numpy(), np.ones((len(dataset), 1))])
    targets_np = dataset["Therapeutic Dose of Warfarin"].to_numpy()
    return features_np, targets_np


def _initialize_linear_model(
    X: np.ndarray, y: np.ndarray
) -> linear_model.LinearRegression:
    # Randomly initialize the linear model. The model cannot be used to predict
    # without being fitted on something first.
    random_x = np.random.rand(*X.shape)
    random_y = np.random.rand(*y.shape)
    model = linear_model.LinearRegression().fit(random_x, random_y)
    return model
