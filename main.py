import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from preprocessing import preprocess_patients_df
from dosing_algo import fixed_dose_policy, pharmacogenetic_policy, linucb_policy


def compute_metrics(dosage_bucket, dosage_target):
    n_patients = len(dosage_bucket)
    regrets = []
    incorrect_fracs = []
    total_regret = 0
    for i in range(n_patients):
        is_correct = 0
        if (
            (dosage_bucket[i] == 0 and dosage_target[i] < 21)
            or (
                dosage_bucket[i] == 1
                and dosage_target[i] >= 21
                and dosage_target[i] <= 49
            )
            or (dosage_bucket[i] == 2 and dosage_target[i] > 49)
        ):
            is_correct = 1
        total_regret += 1 - is_correct
        regrets.append(total_regret)
        incorrect_fracs.append(total_regret / (i + 1))
    return regrets, np.array(incorrect_fracs)


def plot_performance(
    fixed_dose_metric, lin_oracle_metric, linucb_metric, metric_name, file_name
):
    n_patients = fixed_dose_metric.shape[1]
    timesteps = np.linspace(0, n_patients - 1, n_patients)

    fixed_dose_mean = fixed_dose_metric.mean(axis=0)
    lin_oracle_mean = lin_oracle_metric.mean(axis=0)
    linucb_mean = linucb_metric.mean(axis=0)
    fixed_dose_ci = 1.96 * fixed_dose_metric.std(axis=0)
    lin_oracle_ci = 1.96 * lin_oracle_metric.std(axis=0)
    linucb_ci = 1.96 * linucb_metric.std(axis=0)

    plt.plot(timesteps, fixed_dose_mean, label="Fixed Dose")
    plt.plot(timesteps, lin_oracle_mean, label="Pharmocogenetic Dose")
    plt.plot(timesteps, linucb_mean, label="LinUCB Dose")
    plt.fill_between(
        timesteps,
        (fixed_dose_mean - fixed_dose_ci),
        (fixed_dose_mean + fixed_dose_ci),
        color="blue",
        alpha=0.1,
    )
    plt.fill_between(
        timesteps,
        (lin_oracle_mean - lin_oracle_ci),
        (lin_oracle_mean + lin_oracle_ci),
        color="orange",
        alpha=0.1,
    )
    plt.fill_between(
        timesteps,
        (linucb_mean - linucb_ci),
        (linucb_mean + linucb_ci),
        color="green",
        alpha=0.1,
    )
    plt.xlabel("Number of patients")
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(file_name)
    plt.clf()


if __name__ == "__main__":
    np.random.seed(42)

    patients_df = pd.read_csv("data/warfarin.csv")
    patients_df = preprocess_patients_df(patients_df)

    N_TRIALS = 20
    fixed_dose_regret_trials, fixed_dose_incorrect_frac_trials = [], []
    linear_oracle_regret_trials, linear_oracle_incorrect_frac_trials = [], []
    linucb_regret_trials, linucb_incorrect_frac_trials = [], []
    for _ in range(N_TRIALS):
        shuffled_patients_df = patients_df.sample(frac=1).reset_index(drop=True)
        dosage_target = shuffled_patients_df["Therapeutic Dose of Warfarin"].to_numpy()

        fixed_dose_bucket = fixed_dose_policy(shuffled_patients_df)
        fixed_dose_regret, fixed_dose_incorrect_frac = compute_metrics(
            fixed_dose_bucket, dosage_target
        )
        fixed_dose_regret_trials.append(fixed_dose_regret)
        fixed_dose_incorrect_frac_trials.append(fixed_dose_incorrect_frac)

        linear_oracle_bucket = pharmacogenetic_policy(shuffled_patients_df)
        linear_oracle_regret, linear_oracle_incorrect_frac = compute_metrics(
            linear_oracle_bucket, dosage_target
        )
        linear_oracle_regret_trials.append(linear_oracle_regret)
        linear_oracle_incorrect_frac_trials.append(linear_oracle_incorrect_frac)

        linucb_dose_bucket = linucb_policy(shuffled_patients_df)
        linucb_regret, linucb_incorrect_frac = compute_metrics(
            linucb_dose_bucket, dosage_target
        )
        linucb_regret_trials.append(linucb_regret)
        linucb_incorrect_frac_trials.append(linucb_incorrect_frac)

    plot_performance(
        np.array(fixed_dose_regret_trials),
        np.array(linear_oracle_regret_trials),
        np.array(linucb_regret_trials),
        metric_name="Regret",
        file_name="regret.png",
    )

    ax = plt.gca()
    ax.set_ylim([0.2, 0.8])
    plot_performance(
        np.array(fixed_dose_incorrect_frac_trials),
        np.array(linear_oracle_incorrect_frac_trials),
        np.array(linucb_incorrect_frac_trials),
        metric_name="Fraction of incorrect dosing decision",
        file_name="incorrect_fraction.png",
    )
