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


def contextual_mab_experiments(patients_df):
    N_TRIALS = 2
    # smaller bound than paper
    linucb_sm1_regret_trials, linucb_sm1_incorrect_frac_trials = [], []
    # smaller bound than paper
    linucb_sm2_regret_trials, linucb_sm2_incorrect_frac_trials = [], []
    # delta = 0.01
    linucb_1_regret_trials, linucb_1_incorrect_frac_trials = [], []
    # delta = 0.10
    linucb_10_regret_trials, linucb_10_incorrect_frac_trials = [], []
    # smaller bound than paper
    oful_sm_regret_trials, oful_sm_incorrect_frac_trials = [], []
    # delta = 0.01
    oful_1_regret_trials, oful_1_incorrect_frac_trials = [], []
    # delta = 0.10
    oful_10_regret_trials, oful_10_incorrect_frac_trials = [], []

    for i in range(N_TRIALS):
        print("Starting trial ", i)
        shuffled_patients_df = patients_df.sample(frac=1).reset_index(drop=True)
        dosage_target = shuffled_patients_df["Therapeutic Dose of Warfarin"].to_numpy()

        # linucb_sm1_dose_bucket = linucb_policy(shuffled_patients_df, beta=0.10)
        # linucb_sm1_regret, linucb_sm1_incorrect_frac = compute_metrics(
        #     linucb_sm1_dose_bucket, dosage_target
        # )
        # linucb_sm1_regret_trials.append(linucb_sm1_regret)
        # linucb_sm1_incorrect_frac_trials.append(linucb_sm1_incorrect_frac)

        # linucb_sm2_dose_bucket = linucb_policy(shuffled_patients_df, beta=0.50)
        # linucb_sm2_regret, linucb_sm2_incorrect_frac = compute_metrics(
        #     linucb_sm2_dose_bucket, dosage_target
        # )
        # linucb_sm2_regret_trials.append(linucb_sm2_regret)
        # linucb_sm2_incorrect_frac_trials.append(linucb_sm2_incorrect_frac)

        # linucb_1_dose_bucket = linucb_policy(shuffled_patients_df, beta=2.22)
        # linucb_1_regret, linucb_1_incorrect_frac = compute_metrics(
        #     linucb_1_dose_bucket, dosage_target
        # )
        # linucb_1_regret_trials.append(linucb_1_regret)
        # linucb_1_incorrect_frac_trials.append(linucb_1_incorrect_frac)

        # linucb_10_dose_bucket = linucb_policy(shuffled_patients_df, beta=2.63)
        # linucb_10_regret, linucb_10_incorrect_frac = compute_metrics(
        #     linucb_10_dose_bucket, dosage_target
        # )
        # linucb_10_regret_trials.append(linucb_10_regret)
        # linucb_10_incorrect_frac_trials.append(linucb_10_incorrect_frac)

        oful_sm_dose_bucket = oful_policy(
            shuffled_patients_df, var=0.02, delta=0.10, S=0.4
        )
        oful_sm_regret, oful_sm_incorrect_frac = compute_metrics(
            oful_sm_dose_bucket, dosage_target
        )
        oful_sm_regret_trials.append(oful_sm_regret)
        oful_sm_incorrect_frac_trials.append(oful_sm_incorrect_frac)

        # oful_1_dose_bucket = oful_policy(shuffled_patients_df, delta=0.01)
        # oful_1_regret, oful_1_incorrect_frac = compute_metrics(
        #     oful_1_dose_bucket, dosage_target
        # )
        # oful_1_regret_trials.append(oful_1_regret)
        # oful_1_incorrect_frac_trials.append(oful_1_incorrect_frac)

        # oful_10_dose_bucket = oful_policy(shuffled_patients_df, delta=0.10)
        # oful_10_regret, oful_10_incorrect_frac = compute_metrics(
        #     oful_10_dose_bucket, dosage_target
        # )
        # oful_10_regret_trials.append(oful_10_regret)
        # oful_10_incorrect_frac_trials.append(oful_10_incorrect_frac)

    # print('LinUCB (beta = 0.1): ', np.mean(np.array(linucb_sm1_incorrect_frac_trials)[:, -1]))
    # print('LinUCB (beta = 0.5): ', np.mean(np.array(linucb_sm2_incorrect_frac_trials)[:, -1]))
    # print('LinUCB (delta = 0.01): ', np.mean(np.array(linucb_1_incorrect_frac_trials)[:, -1]))
    # print('LinUCB (delta = 0.1): ', np.mean(np.array(linucb_10_incorrect_frac_trials)[:, -1]))
    print("OFUL (small): ", np.mean(np.array(oful_sm_incorrect_frac_trials)[:, -1]))
    # print('OFUL (delta = 0.01): ', np.mean(np.array(oful_1_incorrect_frac_trials)[:, -1]))
    # print('OFUL (delta = 0.1): ', np.mean(np.array(oful_10_incorrect_frac_trials)[:, -1]))

    # print('OFUL (delta = 0.01): ', np.mean(oful_1_incorrect_frac_trials[:, -1]))
    # print('OFUL (delta = 0.1): ', np.mean(oful_10_incorrect_frac_trials[:, -1]))

    # ax = plt.gca()
    # ax.set_ylim([0.2, 0.8])
    # plot_performance(
    #     metrics=[
    #         Metric(
    #             metric_values=linucb_sm_incorrect_frac_trials,
    #             label_name=r"LinUCB ($\beta=0.5)",
    #             color="pink",
    #         ),
    #         Metric(
    #             metric_values=linucb_1_incorrect_frac_trials,
    #             label_name=r"LinUCB ($\delta=0.01)",
    #             color="red",
    #         ),
    #         Metric(
    #             metric_values=linucb_10_incorrect_frac_trials,
    #             label_name=r"LinUCB ($\delta=0.1)",
    #             color="maroon",
    #         ),
    #         Metric(
    #             metric_values=oful_1_incorrect_frac_trials,
    #             label_name=r"LinUCB ($\delta=0.01)",
    #             color="green",
    #         ),
    #         Metric(
    #             metric_values=oful_1_incorrect_frac_trials,
    #             label_name=r"LinUCB ($\delta=0.01)",
    #             color="purple",
    #         ),
    #     ],
    #     n_steps=np.array(fixed_dose_incorrect_frac_trials).shape[1],
    #     metric_name="Fraction of incorrect dosing decision",
    #     file_name="incorrect_fraction.png",
    # )


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
        metric_name="CLinUCB",
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


def compare_all_methods(patients_df: pd.DataFrame) -> None:
    N_TRIALS = 20

    regret_by_method_names = collections.defaultdict(list)
    incorrect_frac_by_method_names = collections.defaultdict(list)

    for i in range(N_TRIALS):
        print(f"Running trial {i}")
        shuffled_patients_df = patients_df.sample(frac=1).reset_index(drop=True)
        dosage_target = shuffled_patients_df["Therapeutic Dose of Warfarin"].to_numpy()

        for method_name, config in _METHOD_NAME_TO_ALGOS.items():
            buckets = config.algo(shuffled_patients_df)
            regret, incorrect_frac = compute_metrics(buckets, dosage_target)
            regret_by_method_names[method_name].append(regret)
            incorrect_frac_by_method_names[method_name].append(incorrect_frac)

            regret_df = pd.DataFrame(data=regret_by_method_names[method_name])
            incorrect_frac_df = pd.DataFrame(
                data=incorrect_frac_by_method_names[method_name]
            )
            regret_df.to_csv(f"output/regret-{method_name}.csv")
            incorrect_frac_df.to_csv(f"output/incorrect-frac-{method_name}.csv")

    plot_performance(
        metrics=[
            Metric(
                metric_values=regret_by_method_names[method_name],
                label_name=config.metric_name,
                color=config.color,
            )
            for method_name, config in _METHOD_NAME_TO_ALGOS.items()
        ],
        n_steps=np.array(regret_by_method_names["fixed-dose"]).shape[1],
        metric_name="Regret",
        file_name="regret.png",
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
            for method_name, config in _METHOD_NAME_TO_ALGOS.items()
        ],
        n_steps=np.array(incorrect_frac_by_method_names["fixed-dose"]).shape[1],
        metric_name="Fraction of incorrect dosing decisions",
        file_name="incorrect_fraction.png",
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


def safe_lin_ucb_experiment(patients_df):
    N_TRIALS = 20

    linucb_beta20_trials = []
    linucb_beta05_trials = []
    safeucb_beta20_alpha10_trials = []
    safeucb_beta05_alpha10_trials = []
    fixed_dose_trials = []
    for _ in range(N_TRIALS):
        shuffled_patients_df = patients_df.sample(frac=1).reset_index(drop=True)[0:1000]
        dosage_target = shuffled_patients_df["Therapeutic Dose of Warfarin"].to_numpy()

        linucb_beta20_bucket = linucb_policy(shuffled_patients_df, beta=4.0)
        _, linucb_beta20_incorrect = compute_metrics(
            linucb_beta20_bucket, dosage_target
        )
        linucb_beta20_trials.append(linucb_beta20_incorrect)

        linucb_beta05_bucket = linucb_policy(shuffled_patients_df, beta=0.5)
        _, linucb_beta05_incorrect = compute_metrics(
            linucb_beta05_bucket, dosage_target
        )
        linucb_beta05_trials.append(linucb_beta05_incorrect)

        safeucb_beta20_alpha10_bucket = safe_linucb_policy(
            shuffled_patients_df, alpha=0.10, beta=4.0
        )
        _, safeucb_beta20_incorrect = compute_metrics(
            safeucb_beta20_alpha10_bucket, dosage_target
        )
        safeucb_beta20_alpha10_trials.append(safeucb_beta20_incorrect)

        safeucb_beta05_alpha10_bucket = safe_linucb_policy(
            shuffled_patients_df, alpha=0.10, beta=0.5
        )
        _, safeucb_beta05_incorrect = compute_metrics(
            safeucb_beta05_alpha10_bucket, dosage_target
        )
        safeucb_beta05_alpha10_trials.append(safeucb_beta05_incorrect)

        fixed_dose_bucket = fixed_dose_policy(shuffled_patients_df)
        _, fixed_dose_incorrect = compute_metrics(fixed_dose_bucket, dosage_target)
        fixed_dose_trials.append(fixed_dose_incorrect)

    ax = plt.gca()
    ax.set_ylim([0.2, 0.8])

    plot_performance(
        metrics=[
            Metric(
                metric_values=fixed_dose_trials,
                label_name="Fixed Dose",
                color="blue",
            ),
            Metric(
                metric_values=linucb_beta05_trials,
                label_name=r"LinUCB",
                color="red",
            ),
            Metric(
                metric_values=safeucb_beta05_alpha10_trials,
                label_name=r"CLUCB",
                color="teal",
            ),
        ],
        n_steps=np.array(fixed_dose_trials).shape[1],
        metric_name="Fraction of incorrect dosing decision",
        file_name="incorrect_frac_initial.png",
    )


if __name__ == "__main__":
    np.random.seed(42)

    patients_df = pd.read_csv("data/warfarin.csv")
    patients_df = preprocess_patients_df(patients_df)

    # compare_ucb_methods(patients_df)
    # contextual_mab_experiments(patients_df)
    compare_all_methods(patients_df)
    # safe_lin_ucb_experiment(patients_df)
