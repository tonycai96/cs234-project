import collections
import dataclasses
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import supervised_learning
import thompson_sampling
from dosing_algo import (fixed_dose_policy, gaussian_ts_policy, linucb_policy,
                         oful_policy, pharmacogenetic_policy,
                         safe_linucb_policy)
from preprocessing import preprocess_patients_df


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
            or (dosage_bucket[i] == 1 and 21 <= dosage_target[i] <= 49)
            or (dosage_bucket[i] == 2 and dosage_target[i] > 49)
        ):
            is_correct = 1
        total_regret += 1 - is_correct
        regrets.append(total_regret)
        incorrect_fracs.append(total_regret / (i + 1))
    return regrets, np.array(incorrect_fracs)


def plot_performance(
    metrics: list[Metric],
    n_steps: int,
    metric_name: str,
    file_name: str,
) -> None:
    timesteps = np.linspace(0, n_steps - 1, n_steps)

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


def compare_ucb_methods(patients_df):
    N_TRIALS = 2

    linucb_regret_trials, linucb_incorrect_frac_trials = [], []
    ts_regret_trials, ts_incorrect_frac_trials = [], []
    clucb_regret_trials, clucb_incorrec_frac_trials = [], []

    for i in range(N_TRIALS):
        print("Starting trial ", i)
        shuffled_patients_df = patients_df.sample(frac=1).reset_index(drop=True)
        dosage_target = shuffled_patients_df["Therapeutic Dose of Warfarin"].to_numpy()

        linucb_dose_bucket = linucb_policy(shuffled_patients_df, beta=0.5)
        linucb_regret, linucb_incorrect_frac = compute_metrics(
            linucb_dose_bucket, dosage_target
        )
        linucb_regret_trials.append(linucb_regret)
        linucb_incorrect_frac_trials.append(linucb_incorrect_frac)

        ts_dose_bucket = gaussian_ts_policy(shuffled_patients_df, var=0.5)
        ts_regret, ts_incorrect_frac = compute_metrics(ts_dose_bucket, dosage_target)
        ts_regret_trials.append(ts_regret)
        ts_incorrect_frac_trials.append(ts_incorrect_frac)

    print("LinUCB = ", np.mean(np.array(linucb_incorrect_frac_trials)[:, -1]))
    print("TS (v = 1) = ", np.mean(np.array(ts_incorrect_frac_trials)[:, -1]))


@dataclasses.dataclass(frozen=True)
class AlgoConfig:
    algo: Callable
    metric_name: str
    color: str


_METHOD_NAME_TO_ALGOS = {
    "fixed-dose": AlgoConfig(
        algo=fixed_dose_policy, metric_name="Fixed Dose", color="blue"
    ),
    "pharmacogenetic": AlgoConfig(
        algo=pharmacogenetic_policy, metric_name="Pharmacogenetic", color="red"
    ),
    "lin-ucb": AlgoConfig(
        algo=lambda df: linucb_policy(df, beta=0.5), metric_name="LinUCB", color="green"
    ),
    "cl-ucb": AlgoConfig(
        algo=lambda df: safe_linucb_policy(df, alpha=0.1, beta=0.5),
        metric_name="CLUCB",
        color="brown",
    ),
    "bandit-sl": AlgoConfig(
        algo=supervised_learning.bandit_sl, metric_name="Bandit SL", color="gold"
    ),
    "beta-ts": AlgoConfig(
        algo=thompson_sampling.thompson_sampling, metric_name="Beta TS", color="purple"
    ),
    "gaussian-ts": AlgoConfig(
        algo=lambda df: gaussian_ts_policy(df, var=0.5),
        metric_name="Gaussian TS",
        color="gray",
    ),
}

_TS_ALGOS = {
    "lin-ucb": AlgoConfig(
        algo=lambda df: linucb_policy(df, beta=0.5), metric_name="LinUCB", color="green"
    ),
    "gaussian-ts-0.5": AlgoConfig(
        algo=lambda df: gaussian_ts_policy(df, var=0.5),
        metric_name="Gaussian TS (v=0.5)",
        color="green",
    ),
    "gaussian-ts-0.3": AlgoConfig(
        algo=lambda df: gaussian_ts_policy(df, var=0.3),
        metric_name="Gaussian TS (v=0.3)",
        color="purple",
    ),
    "gaussian-ts-0.1": AlgoConfig(
        algo=lambda df: gaussian_ts_policy(df, var=0.3),
        metric_name="Gaussian TS (v=0.1)",
        color="red",
    ),
}

_ROBUSTNESS_ALGOS = {
    "lin-ucb": AlgoConfig(
        algo=lambda df: linucb_policy(df, beta=0.5), metric_name="LinUCB", color="teal"
    ),
    "gaussian-ts-0.3": AlgoConfig(
        algo=lambda df: gaussian_ts_policy(df, var=0.3),
        metric_name="Gaussian TS (v=0.3)",
        color="brown",
    ),
    "fixed-dose": AlgoConfig(
        algo=fixed_dose_policy, metric_name="Fixed Dose", color="gray"
    ),
    "cl-ucb": AlgoConfig(
        algo=lambda df: safe_linucb_policy(df, alpha=0.1, beta=0.5),
        metric_name="CLUCB",
        color="blue",
    ),
}


_ALGOS_PARAMS = {
    "lin-ucb1": AlgoConfig(
        algo=lambda df: linucb_policy(df, beta=0.5), metric_name="LinUCB", color="green"
    ),
    "lin-ucb2": AlgoConfig(
        algo=lambda df: linucb_policy(df, beta=2.22), metric_name="LinUCB", color="green"
    ),
    "lin-ucb3": AlgoConfig(
        algo=lambda df: linucb_policy(df, beta=2.63), metric_name="LinUCB", color="green"
    ),
}


def run_experiments(patients_df: pd.DataFrame, algos_dict: list, exp_name: str) -> None:
    regret_by_method_names = collections.defaultdict(list)
    incorrect_frac_by_method_names = collections.defaultdict(list)

    N_TRIALS = 20
    for i in range(N_TRIALS):
        print(f"Running trial {i}")
        shuffled_patients_df = patients_df.sample(frac=1).reset_index(drop=True)
        dosage_target = shuffled_patients_df["Therapeutic Dose of Warfarin"].to_numpy()

        for method_name, config in algos_dict.items():
            buckets = config.algo(shuffled_patients_df)
            regret, incorrect_frac = compute_metrics(buckets, dosage_target)
            regret_by_method_names[method_name].append(regret)
            incorrect_frac_by_method_names[method_name].append(incorrect_frac)

            regret_df = pd.DataFrame(data=regret_by_method_names[method_name])
            incorrect_frac_df = pd.DataFrame(
                data=incorrect_frac_by_method_names[method_name]
            )
            regret_df.to_csv(f"output/{exp_name}-regret-{method_name}.csv")
            incorrect_frac_df.to_csv(f"output/{exp_name}-incorrect-frac-{method_name}.csv")

    n_steps = len(patients_df)
    plot_performance(
        metrics=[
            Metric(
                metric_values=regret_by_method_names[method_name],
                label_name=config.metric_name,
                color=config.color,
            )
            for method_name, config in algos_dict.items()
        ],
        n_steps=n_steps,
        metric_name="Regret",
        file_name="{0}_regret.png".format(exp_name),
    )

    ax = plt.gca()
    ax.set_ylim([0.2, 0.8])
    plot_performance(
        metrics=[
            Metric(
                metric_values=incorrect_frac_by_method_names[method_name],
                label_name=config.metric_name,
                color=config.color,
            )
            for method_name, config in algos_dict.items()
        ],
        n_steps=n_steps,
        metric_name="Fraction of incorrect dosing decisions",
        file_name="{0}_incorrect_fraction.png".format(exp_name),
    )
    final_cumulative_regrets = {
        method_name: np.average([r[-1] for r in regrets])
        for method_name, regrets in regret_by_method_names.items()
    }
    final_incorrect_fracs = {
        method_name: np.average([f[-1] for f in fracs])
        for method_name, fracs in incorrect_frac_by_method_names.items()
    }
    print(f"Average Cumulative Regrets over Trials: {final_cumulative_regrets}")
    print(f"Average Incorrect Fractions over Trials: {final_incorrect_fracs}")


if __name__ == "__main__":
    np.random.seed(42)

    patients_df = pd.read_csv("data/warfarin.csv")
    patients_df = preprocess_patients_df(patients_df)

    run_experiments(patients_df, _METHOD_NAME_TO_ALGOS, exp_name="all")
    # run_experiments(patients_df, _TS_ALGOS)
    # run_experiments(patients_df[0:1000], _ROBUSTNESS_ALGOS, exp_name="safety_trials")
    # run_experiments(patients_df, _ALGOS_PARAMS, exp_name="hyperparam_tuning")
    # safe_lin_ucb_experiment(patients_df)
