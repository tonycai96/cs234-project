from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from preprocessing import preprocess_patients_df
from dosing_algo import fixed_dose_policy, pharmacogenetic_policy, linucb_policy


def compute_metrics(dosage_bucket, dosage_target):
    n_patients = len(dosage_bucket)
    regrets = []
    incorrect_frac = []
    total_regret = 0
    for i in range(n_patients):
        is_correct = 0
        if (dosage_bucket[i] == 0 and dosage_target[i] < 21) or (dosage_bucket[i] == 1 and dosage_target[i] >= 21 and dosage_target[i] <= 49) or (dosage_bucket[i] == 2 and dosage_target[i] > 49):
            is_correct = 1
        total_regret += is_correct
        regrets.append(total_regret)
        incorrect_frac.append(1.0 - total_regret / (i+1))
    return regrets, incorrect_frac


def plot_performance(fixed_regret, lin_oracle_regret, linucb_regret):
    pass


if __name__ == "__main__":
    np.random.seed(42)

    patients_df = pd.read_csv("data/warfarin.csv")
    patients_df = preprocess_patients_df(patients_df)

    N_TRIALS = 1
    for _ in range(N_TRIALS):
        shuffled_patients_df = patients_df.sample(frac=1).reset_index(drop=True)
        dosage_target = shuffled_patients_df["Therapeutic Dose of Warfarin"].to_numpy()

        fixed_dose_bucket = fixed_dose_policy(shuffled_patients_df)
        fixed_regret, fixed_incorrect_frac = compute_metrics(fixed_dose_bucket, dosage_target)

        linear_oracle_bucket = pharmacogenetic_policy(shuffled_patients_df)
        linear_oracle_regret, linear_oracle_incorrect_frac = compute_metrics(linear_oracle_bucket, dosage_target)

        linucb_dose_bucket = linucb_policy(shuffled_patients_df)
        linucb_regret, linucb_incorrect_frac = compute_metrics(linucb_dose_bucket, dosage_target)

    print(fixed_incorrect_frac[-1])
    print(linear_oracle_incorrect_frac[-1])
    print(linucb_incorrect_frac[-1])

    plot_performance(fixed_regret, linear_oracle_regret, linucb_regret)
    plot_performance(fixed_regret, linear_oracle_regret, linucb_regret)

    # incorrect_dose_frac = []
    # n_incorrect = 0
    # for i in range(1, n_patients+1):
    #     n_incorrect -= arms_rewards[i-1, 1]
    #     print(arms_rewards[i-1, 1])
    #     incorrect_dose_frac.append(n_incorrect / i)

    # timesteps = np.linspace(1, n_patients, n_patients)
    # plt.plot(timesteps, incorrect_dose_frac)
    # plt.show()
