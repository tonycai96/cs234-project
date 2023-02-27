import pandas as pd
import torch
import numpy


TARGET_COL = "Therapeutic Dose of Warfarin"


def fixed_actions(dataset):
    correct = warfarin_df[
        (warfarin_df["Therapeutic Dose of Warfarin"] >= 21)
        & (warfarin_df["Therapeutic Dose of Warfarin"] <= 49)]
    return 1.0 * len(correct) / len(dataset)


if __name__ == "__main__":
    warfarin_df = pd.read_csv('data/warfarin.csv')
    accuracy = fixed_actions(warfarin_df)
    print('Fixed dose accuracy = ', accuracy)
