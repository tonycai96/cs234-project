from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from linucb import linear_ucb
from preprocessing import preprocess_patients_df


def fixed_actions(dataset: pd.DataFrame) -> float:
    correct = dataset[
        (dataset["Therapeutic Dose of Warfarin"] >= 21)
        & (dataset["Therapeutic Dose of Warfarin"] <= 49)
    ]
    x = dataset["Therapeutic Dose of Warfarin"].to_numpy() - 35
    x = x**2
    print('mse (fixed dose) = ', x.mean())
    return 1.0 * len(correct) / len(dataset)


def clinical_dose(dataset: pd.DataFrame) -> float:
    dosage_df = dataset.copy()
    dosage_df["Predicted Dose"] = (
        4.0376
        - 0.2546 * dataset["_age_decade"]
        + 0.0118 * dataset["Height (cm)"]
        + 0.0134 * dataset["Weight (kg)"]
        - 0.6752 * dataset["_race_asian"]
        + 0.4060 * dataset["_race_black"]
        + 0.0443 * dataset["_race_other"]
        - 0.5695 * dataset["Amiodarone (Cordarone)"]
    )
    enzyme_inducer_active = (
        (dosage_df["Carbamazepine (Tegretol)"] == 1.0)
        | (dosage_df["Phenytoin (Dilantin)"] == 1.0)
        | (dosage_df["Rifampin or Rifampicin"] == 1.0)
    )
    dosage_df.loc[enzyme_inducer_active, "Predicted Dose"] += 1.2799
    dosage_df["Predicted Dose"] = dosage_df["Predicted Dose"] ** 2

    low_dose = (dosage_df["Predicted Dose"] < 21) & (
        dosage_df["Therapeutic Dose of Warfarin"] < 21
    )
    mid_dose = (
        (dosage_df["Predicted Dose"] >= 21)
        & (dosage_df["Therapeutic Dose of Warfarin"] >= 21)
    ) & (
        (dosage_df["Predicted Dose"] <= 49)
        & (dosage_df["Therapeutic Dose of Warfarin"] <= 49)
    )
    high_dose = (dosage_df["Predicted Dose"] > 49) & (
        dosage_df["Therapeutic Dose of Warfarin"] > 49
    )
    correct = dosage_df[low_dose | mid_dose | high_dose]
    return 1.0 * len(correct) / len(dataset)


def pharmacogenetic_dose(dataset: pd.DataFrame) -> float:
    dosage_df = dataset.copy()
    dosage_df["Predicted Dose"] = (
        5.6044
        - 0.2614 * dataset["_age_decade"]
        + 0.0087 * dataset["Height (cm)"]
        + 0.0128 * dataset["Weight (kg)"]
        - 0.8677 * dataset["VKORC1 A/G"]
        - 1.6974 * dataset["VKORC1 A/A"]
        - 0.4854 * dataset["VKORC1 unknown"]
        - 0.5211 * dataset["CYP2C9 *1/*2"]
        - 0.9357 * dataset["CYP2C9 *1/*3"]
        - 1.0616 * dataset["CYP2C9 *2/*2"]
        - 1.9206 * dataset["CYP2C9 *2/*3"]
        - 2.3312 * dataset["CYP2C9 *3/*3"]
        - 0.2188 * dataset["CYP2C9 unknown"]
        - 0.1092 * dataset["_race_asian"]
        - 0.2760 * dataset["_race_black"]
        - 0.1032 * dataset["_race_other"]
        - 0.5503 * dataset["Amiodarone (Cordarone)"]
    )
    enzyme_inducer_active = (
        (dosage_df["Carbamazepine (Tegretol)"] == 1.0)
        | (dosage_df["Phenytoin (Dilantin)"] == 1.0)
        | (dosage_df["Rifampin or Rifampicin"] == 1.0)
    )
    dosage_df.loc[enzyme_inducer_active, "Predicted Dose"] += 1.1816
    dosage_df["Predicted Dose"] = dosage_df["Predicted Dose"] ** 2

    low_dose = (dosage_df["Predicted Dose"] < 21) & (
        dosage_df["Therapeutic Dose of Warfarin"] < 21
    )
    mid_dose = (
        (dosage_df["Predicted Dose"] >= 21)
        & (dosage_df["Therapeutic Dose of Warfarin"] >= 21)
    ) & (
        (dosage_df["Predicted Dose"] <= 49)
        & (dosage_df["Therapeutic Dose of Warfarin"] <= 49)
    )
    high_dose = (dosage_df["Predicted Dose"] > 49) & (
        dosage_df["Therapeutic Dose of Warfarin"] > 49
    )
    correct = dosage_df[low_dose | mid_dose | high_dose]

    pred_dose = dosage_df["Predicted Dose"].to_numpy()
    real_dose = dosage_df["Therapeutic Dose of Warfarin"].to_numpy()
    x = (pred_dose - real_dose)**2
    print('mse (pharmocogenetic) = ', x.mean())

    return 1.0 * len(correct) / len(dataset)


def compute_features_and_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset ordering

    n_samples, n_arms = len(df), 3

    features_df = df.drop("Therapeutic Dose of Warfarin", axis=1)
    features_np = np.hstack([features_df.to_numpy(), np.ones((n_samples, 1))])

    dosage_np = df["Therapeutic Dose of Warfarin"].to_numpy()
    targets_np = np.zeros((n_samples, n_arms))
    targets_np = np.zeros((n_samples, n_arms))
    targets_np[dosage_np < 21, 1] = -1.0
    targets_np[dosage_np < 21, 2] = -1.0
    targets_np[np.all([dosage_np >= 21, dosage_np <= 49], axis=0), 0] = -1.0
    targets_np[np.all([dosage_np >= 21, dosage_np <= 49], axis=0), 2] = -1.0
    targets_np[dosage_np > 49, 0] = -1.0
    targets_np[dosage_np > 49, 1] = -1.0
    return features_np, targets_np


if __name__ == "__main__":
    np.random.seed(12)

    patients_df = pd.read_csv("data/warfarin.csv")
    processed_df = preprocess_patients_df(patients_df)

    # Part (1)
    fixed_dose_acc = fixed_actions(processed_df)
    print(f"Fixed dose accuracy = {fixed_dose_acc}")
    clinical_dose_acc = clinical_dose(processed_df)
    print(f"Clinical dose accuracy = {clinical_dose_acc}")
    pharmacogenetic_dose_acc = pharmacogenetic_dose(processed_df)
    print(f"Pharmacogenetic dose accuracy = {pharmacogenetic_dose_acc}")

    # Part (2)
    features, arms_rewards = compute_features_and_targets(processed_df)
    linucb_dose = linear_ucb(features, arms_rewards, alpha=1)

    n_patients = len(features)
    regrets = [0.0]
    num_correct_preds = 0
    for i in range(n_patients):
        regrets.append(regrets[-1] - min(0.0, arms_rewards[i][linucb_dose[i]]))
    print(f"LinUCB final accuracy = {1.0 - regrets[-1] / n_patients}")

    # incorrect_dose_frac = []
    # n_incorrect = 0
    # for i in range(1, n_patients+1):
    #     n_incorrect -= arms_rewards[i-1, 1]
    #     print(arms_rewards[i-1, 1])
    #     incorrect_dose_frac.append(n_incorrect / i)

    # timesteps = np.linspace(1, n_patients, n_patients)
    # plt.plot(timesteps, incorrect_dose_frac)
    # plt.show()
