import dataclasses
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
        self.beta += -reward


class ContextFreeMAB:
    def __init__(self, priors: list[BetaParams]) -> None:
        self.priors = priors
        self.rng = np.random.default_rng(seed=42)

    def play_arm(self, target: np.ndarray[Literal[0, -1]]) -> int:
        chosen_arm = np.argmax(self.rng.beta(a=p.alpha, b=p.beta) for p in self.priors)
        actual_reward = target[chosen_arm]
        self.priors[chosen_arm].update(actual_reward)
        return chosen_arm


def _compute_features_and_targets(
    df: pd.DataFrame,
) -> Tuple[np.ndarray[float], np.ndarray[np.ndarray[int]]]:
    n_samples, n_arms = len(df), 3

    # Discretize all continuous features.
    # Divide height and weights into 3 buckets each.
    df.assign(_is_height_between_120_and_150=lambda x: 120 <= x["Height (cm)"] < 150)
    df.assign(_is_height_between_180_and_210=lambda x: 150 <= x["Height (cm)"] < 180)
    df.assign(_is_height_between_180_and_210=lambda x: 180 <= x["Height (cm)"] < 210)
    df.assign(_is_weight_between_30_and_100=lambda x: 30 <= x["Weight (kg)"] < 100)
    df.assign(_is_weight_between_100_and_170=lambda x: 100 <= x["Weight (kg)"] < 170)
    df.assign(_is_weight_between_170_and_240=lambda x: 170 <= x["Weight (kg)"] < 240)

    features_df = df.drop("Therapeutic Dose of Warfarin", axis=1)
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
