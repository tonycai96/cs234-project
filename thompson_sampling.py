import collections
from typing import Literal, Tuple

import numpy as np
import pandas as pd


class BetaParams:
    def __init__(self):
        self.alpha = 1
        self.beta = 1

    def update(self, reward: Literal[0, -1]) -> None:
        # 0 and -1 correspond to 1 and 0 in the original reward scheme, respectively.
        self.alpha += reward + 1
        self.beta -= reward

    def __repr__(self) -> str:
        return f"(alpha: {self.alpha}, beta: {self.beta})"


class ContextFreeMAB:
    def __init__(self, num_arms: int = 3) -> None:
        self.priors = [BetaParams() for _ in range(num_arms)]
        self.rng = np.random.default_rng()

    def play_arm(self, target: np.ndarray[Literal[0, -1]]) -> int:
        rewards = [self.rng.beta(a=p.alpha, b=p.beta) for p in self.priors]
        chosen_arm = np.argmax(rewards)
        actual_reward = target[chosen_arm]
        self.priors[chosen_arm].update(actual_reward)
        return chosen_arm

    def __repr__(self) -> str:
        return str(self.priors)


def thompson_sampling(dataset: pd.DataFrame) -> np.ndarray[int]:
    dosage_buckets = []
    features, targets = _compute_features_and_targets(dataset)
    MABs = _initialize(features)
    for feature, target in zip(features, targets):
        key = _convert_feature_to_bit_string(feature)
        dosage_buckets.append(MABs[key].play_arm(target))

    # print(f"Final MABs: {MABs}")
    return np.array(dosage_buckets)


def _compute_features_and_targets(
    df: pd.DataFrame,
) -> Tuple[np.ndarray[np.ndarray[float]], np.ndarray[np.ndarray[int]]]:
    n_samples, n_arms = len(df), 3

    # Discretize all continuous features.
    # Divide height and weights into 3 buckets.
    df["_is_height_between_120_and_150"] = df["Height (cm)"].between(
        120, 150, inclusive="left"
    )
    df["_is_height_between_150_and_180"] = df["Height (cm)"].between(
        150, 180, inclusive="left"
    )
    df["_is_height_between_180_and_210"] = df["Height (cm)"].between(
        180, 210, inclusive="left"
    )
    df["_is_weight_between_30_and_100"] = df["Weight (kg)"].between(
        30, 100, inclusive="left"
    )
    df["_is_weight_between_100_and_170"] = df["Weight (kg)"].between(
        100, 170, inclusive="left"
    )
    df["_is_weight_between_170_and_240"] = df["Weight (kg)"].between(
        170, 240, inclusive="left"
    )
    # Each decade is in its own bucket and column.
    for decade in range(1, 10, 2):
        df[f"_is_decade_{decade}"] = df["_age_decade"].between(
            decade, decade + 2, inclusive="left"
        )
    features_df = df.copy()
    for column in [
        "Therapeutic Dose of Warfarin",
        "Height (cm)",
        "Weight (kg)",
        "_age_decade",
        "VKORC1 A/G",
        "VKORC1 A/A",
        "VKORC1 unknown",
        "CYP2C9 *1/*2",
        "CYP2C9 *1/*3",
        "CYP2C9 *2/*2",
        "CYP2C9 *2/*3",
        "CYP2C9 *3/*3",
        "CYP2C9 unknown",
    ]:
        features_df = features_df.drop(column, axis=1)
    features_np = np.hstack([features_df.to_numpy(), np.ones((n_samples, 1))])

    dosage_np = df["Therapeutic Dose of Warfarin"].to_numpy()
    targets_np = np.zeros((n_samples, n_arms))
    targets_np[dosage_np < 21, 1] = -1.0
    targets_np[dosage_np < 21, 2] = -1.0
    targets_np[np.all([dosage_np >= 21, dosage_np <= 49], axis=0), 0] = -1.0
    targets_np[np.all([dosage_np >= 21, dosage_np <= 49], axis=0), 2] = -1.0
    targets_np[dosage_np > 49, 0] = -1.0
    targets_np[dosage_np > 49, 1] = -1.0
    return features_np, targets_np


def _initialize(features: np.ndarray[np.ndarray[float]]) -> dict[str, ContextFreeMAB]:
    # Assign one context-free MAB for each combination of features.
    # unique_features_counter = collections.Counter()
    # for f in features:
    #     unique_features_counter[_convert_feature_to_bit_string(f)] += 1
    # print(f"Unique features count: {unique_features_counter}")
    return {_convert_feature_to_bit_string(f): ContextFreeMAB() for f in features}


def _convert_feature_to_bit_string(feature: np.ndarray[float]) -> str:
    return "".join(bit for bit in feature.astype(int).astype(str))
