import numpy as np
import pandas as pd

from linucb import linear_ucb, safe_linear_ucb
from preprocessing import compute_features_and_targets


def fixed_dose_policy(dataset: pd.DataFrame) -> np.ndarray[int]:
    return np.ones(len(dataset))


def pharmacogenetic_policy(dataset: pd.DataFrame) -> np.ndarray[int]:
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

    pred_dose = dosage_df["Predicted Dose"].to_numpy()
    dosage_buckets = np.zeros(len(pred_dose))
    dosage_buckets[np.logical_and(pred_dose >= 21, pred_dose <= 49)] = 1
    dosage_buckets[pred_dose > 49] = 2
    return dosage_buckets


def linucb_policy(dataset: pd.DataFrame, beta: float) -> np.ndarray:
    features, arms_rewards = compute_features_and_targets(dataset)
    return linear_ucb(features, arms_rewards, beta)


def safe_linucb_policy(dataset: pd.DataFrame, alpha: float, beta: float) -> np.ndarray:
    features, arms_rewards = compute_features_and_targets(dataset)
    return safe_linear_ucb(features, arms_rewards, alpha, beta)


if __name__ == "__main__":
    from preprocessing import preprocess_patients_df

    np.random.seed(42)

    patients_df = pd.read_csv("data/warfarin.csv")
    patients_df = preprocess_patients_df(patients_df)

    # # Part (1)
    # fixed_dose_acc = fixed_dose_policy(processed_df)
    # print(f"Fixed dose accuracy = {fixed_dose_acc}")
    # clinical_dose_acc = clinical_dose(processed_df)
    # print(f"Clinical dose accuracy = {clinical_dose_acc}")
    # pharmacogenetic_dose_acc = pharmacogenetic_dose(processed_df)
    # print(f"Pharmacogenetic dose accuracy = {pharmacogenetic_dose_acc}")

    # # Part (2)
    # features, arms_rewards = compute_features_and_targets(processed_df)
    # linucb_dose = linear_ucb(features, arms_rewards, alpha=1)
