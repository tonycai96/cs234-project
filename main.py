import dataclasses
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import supervised_learning
import thompson_sampling
from dosing_algo import (fixed_dose_policy, linucb_policy,
                         pharmacogenetic_policy)
from preprocessing import preprocess_patients_df

N_TRIALS = 20


@dataclasses.dataclass(frozen=True)
class Metric:
    metric_values: list[list[int]] | list[np.ndarray[float]]
    label_name: str
    color: str


def compute_metrics(
    dosage_bucket: np.ndarray, dosage_target: np.ndarray
) -> Tuple[list[int], np.ndarray]:
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
    metrics: list[Metric],
    n_patients: int,
    metric_name: str,
    file_name: str,
) -> None:
    timesteps = np.linspace(0, n_patients - 1, n_patients)

    for metric in metrics:
        metric_values = np.array(metric.metric_values)
        mean = metric_values.mean(axis=0)
        ci = 1.96 * metric_values.std(axis=0)

        plt.plot(timesteps, mean, label=metric.label_name, color=metric.color)
        plt.fill_between(
            timesteps,
            (mean - ci),
            (mean + ci),
            color=metric.color,
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

    fixed_dose_regret_trials, fixed_dose_incorrect_frac_trials = [], []
    linear_oracle_regret_trials, linear_oracle_incorrect_frac_trials = [], []
    linucb_regret_trials, linucb_incorrect_frac_trials = [], []
    bandit_sl_regret_trials, bandit_sl_incorrect_frac_trials = [], []
    ts_regret_trials, ts_incorrect_frac_trials = [], []

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

        bandit_sl_buckets = supervised_learning.bandit_sl(shuffled_patients_df)
        bandit_sl_regret, bandit_sl_incorrect_frac = compute_metrics(
            bandit_sl_buckets, dosage_target
        )
        bandit_sl_regret_trials.append(bandit_sl_regret)
        bandit_sl_incorrect_frac_trials.append(bandit_sl_incorrect_frac)

        ts_buckets = thompson_sampling.thompson_sampling(shuffled_patients_df)
        ts_regret, ts_incorrect_frac = compute_metrics(ts_buckets, dosage_target)
        ts_regret_trials.append(ts_regret)
        ts_incorrect_frac_trials.append(ts_incorrect_frac)

    plot_performance(
        metrics=[
            Metric(
                metric_values=fixed_dose_regret_trials,
                label_name="Fixed Dose",
                color="blue",
            ),
            Metric(
                metric_values=linear_oracle_regret_trials,
                label_name="Pharmacogenetic",
                color="red",
            ),
            Metric(
                metric_values=linucb_regret_trials, label_name="LinUCB", color="green"
            ),
            Metric(
                metric_values=bandit_sl_regret_trials,
                label_name="Bandit SL",
                color="gold",
            ),
            Metric(
                metric_values=ts_regret_trials,
                label_name="Thompson Sampling",
                color="violet",
            ),
        ],
        n_patients=np.array(fixed_dose_regret_trials).shape[1],
        metric_name="Regret",
        file_name="regret.png",
    )

    ax = plt.gca()
    ax.set_ylim([0.2, 0.8])
    plot_performance(
        metrics=[
            Metric(
                metric_values=fixed_dose_incorrect_frac_trials,
                label_name="Fixed Dose",
                color="blue",
            ),
            Metric(
                metric_values=linear_oracle_incorrect_frac_trials,
                label_name="Pharmacogenetic",
                color="red",
            ),
            Metric(
                metric_values=linucb_incorrect_frac_trials,
                label_name="LinUCB",
                color="green",
            ),
            Metric(
                metric_values=bandit_sl_incorrect_frac_trials,
                label_name="Bandit SL",
                color="gold",
            ),
            Metric(
                metric_values=ts_incorrect_frac_trials,
                label_name="Thompson Sampling",
                color="violet",
            ),
        ],
        n_patients=np.array(fixed_dose_incorrect_frac_trials).shape[1],
        metric_name="Fraction of incorrect dosing decision",
        file_name="incorrect_fraction.png",
    )
