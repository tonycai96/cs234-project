import pandas as pd
import numpy as np
from linucb import linear_ucb


TARGET_COL = "Therapeutic Dose of Warfarin"


def fixed_actions(dataset):
    correct = dataset[
        (dataset["Therapeutic Dose of Warfarin"] >= 21)
        & (dataset["Therapeutic Dose of Warfarin"] <= 49)]
    return 1.0 * len(correct) / len(dataset)


def clinical_dose(dataset):
    dosage_df = dataset.copy()
    dosage_df['Predicted Dose'] = 4.0376 + 0.2546*dataset['_age_decade'] + 0.0118*dataset['Height (cm)'] + 0.0134*dataset['Weight (kg)'] \
        - 0.6752*dataset['_race_asian'] + 0.4060*dataset['_race_black'] + 0.0443*dataset['_race_other'] \
        - 0.5695*dataset['Amiodarone (Cordarone)']
    enzyme_inducer_active = (dosage_df['Carbamazepine (Tegretol)'] == 1.0) | (dosage_df['Phenytoin (Dilantin)'] == 1.0) | (dosage_df['Rifampin or Rifampicin'] == 1.0)
    dosage_df.loc[enzyme_inducer_active, 'Predicted Dose'] += 1.2799

    dosage_df['Predicted Dose'] = dosage_df['Predicted Dose']**2
    low_dose = (dosage_df['Predicted Dose'] <= 21) & (dosage_df['Therapeutic Dose of Warfarin'] <= 21)
    mid_dose = ((dosage_df['Predicted Dose'] >= 21) & (dosage_df['Predicted Dose'] >= 21)) \
        & ((dosage_df['Therapeutic Dose of Warfarin'] <= 49) & (dosage_df['Therapeutic Dose of Warfarin'] <= 49))
    high_dose = (dosage_df['Predicted Dose'] > 49) & (dosage_df['Therapeutic Dose of Warfarin'] > 49)
    correct = dosage_df[low_dose | mid_dose | high_dose]
    return 1.0 * len(correct) / len(dataset)


def preprocess_df(old_df):
    columns = [
        'Age',
        'Height (cm)',
        'Weight (kg)',
        'Race',
        'Carbamazepine (Tegretol)',
        'Phenytoin (Dilantin)',
        'Rifampin or Rifampicin',
        'Amiodarone (Cordarone)',
        'Therapeutic Dose of Warfarin']
    
    df = old_df.copy()
    df = df[df['Therapeutic Dose of Warfarin'].notna() & df['Age'].notna()]
    df = df[columns]
    df['_race_asian'] = 0.0
    df['_race_black'] = 0.0
    df['_race_other'] = 0.0
    is_asian = df['Race'] == 'Asian'
    is_black = df['Race'] == 'Black or African American'
    df.loc[is_asian, '_race_asian'] = 1.0
    df.loc[is_black, '_race_black'] = 1.0
    df.loc[~(is_asian | is_black), '_race_other'] = 1.0
    df['_age_decade'] = 0.0
    df = df.assign(_age_decade = lambda x : int(x['Age'][0][0]))
    df = df.drop(columns=['Age', 'Race'])

    df['Height (cm)'].fillna(df['Height (cm)'].mean(numeric_only=True).round(1), inplace=True)
    df['Weight (kg)'].fillna(df['Weight (kg)'].mean(numeric_only=True).round(1), inplace=True)
    df = df.fillna(0.0)
    return df


def compute_features_and_targets(df):
    df = df.sample(frac=1).reset_index(drop=True)
    features_df = df.drop('Therapeutic Dose of Warfarin', axis=1)
    dosage_df = df['Therapeutic Dose of Warfarin']
    n_samples, n_arms = len(features_df), 3
    dosage_np = dosage_df.to_numpy()
    targets_np = np.zeros((n_samples, n_arms))

    features_np = np.hstack([features_df.to_numpy(), np.ones((n_samples, 1))])
    targets_np[dosage_np < 21, 1] = -1.0
    targets_np[dosage_np < 21, 2] = -1.0
    targets_np[np.all([dosage_np >= 21, dosage_np <= 49], axis=0), 0] = -1.0
    targets_np[np.all([dosage_np >= 21, dosage_np <= 49], axis=0), 2] = -1.0
    targets_np[dosage_np > 49, 0] = -1.0
    targets_np[dosage_np > 49, 1] = -1.0
    return features_np, targets_np


if __name__ == "__main__":
    np.random.seed(42)

    patients_df = pd.read_csv('data/warfarin.csv')
    patients_df = preprocess_df(patients_df)

    # Part (1)
    fixed_dose_acc = fixed_actions(patients_df)
    print('Fixed dose accuracy = ', fixed_dose_acc)
    clinical_dose_acc = clinical_dose(patients_df)
    print('Clinical dose accuracy = ', clinical_dose_acc)

    # Part (2)
    features_np, targets_np = compute_features_and_targets(patients_df)
    # print(features_np.shape)
    # print(targets_np.shape)
    # linucb_dose = linear_ucb(features_np, targets_np)
