import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from linucb import linear_ucb


def fixed_actions(dataset):
    correct = dataset[
        (dataset["Therapeutic Dose of Warfarin"] >= 21)
        & (dataset["Therapeutic Dose of Warfarin"] <= 49)]
    return 1.0 * len(correct) / len(dataset)


def clinical_dose(dataset):
    dosage_df = dataset.copy()
    dosage_df['Predicted Dose'] = 4.0376 - 0.2546*dataset['_age_decade'] + 0.0118*dataset['Height (cm)'] + 0.0134*dataset['Weight (kg)'] \
        - 0.6752*dataset['_race_asian'] + 0.4060*dataset['_race_black'] + 0.0443*dataset['_race_other'] \
        - 0.5695*dataset['Amiodarone (Cordarone)']
    enzyme_inducer_active = (dosage_df['Carbamazepine (Tegretol)'] == 1.0) | (dosage_df['Phenytoin (Dilantin)'] == 1.0) | (dosage_df['Rifampin or Rifampicin'] == 1.0)
    dosage_df.loc[enzyme_inducer_active, 'Predicted Dose'] += 1.2799
    dosage_df['Predicted Dose'] = dosage_df['Predicted Dose']**2

    low_dose = (dosage_df['Predicted Dose'] <= 21) & (dosage_df['Therapeutic Dose of Warfarin'] <= 21)
    mid_dose = ((dosage_df['Predicted Dose'] >= 21) & (dosage_df['Therapeutic Dose of Warfarin'] >= 21)) \
        & ((dosage_df['Predicted Dose'] <= 49) & (dosage_df['Therapeutic Dose of Warfarin'] <= 49))
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
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle the dataset ordering

    n_samples, n_arms = len(df), 3

    features_df = df.drop('Therapeutic Dose of Warfarin', axis=1)
    features_np = np.hstack([features_df.to_numpy(), np.ones((n_samples, 1))])

    dosage_np = df['Therapeutic Dose of Warfarin'].to_numpy()
    targets_np = np.ones((n_samples, n_arms))
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
    features, arms_rewards = compute_features_and_targets(patients_df)
    linucb_dose = linear_ucb(features, arms_rewards, alpha=1)

    n_patients = len(features)
    regrets = [0.0]
    for i in range(n_patients):
        regrets.append(regrets[-1] - min(0.0, arms_rewards[i][linucb_dose[i]]))

    timesteps = np.linspace(0, n_patients, n_patients+1)
    plt.plot(timesteps, regrets)
    plt.show()


'''
Index(['PharmGKB Subject ID', 'Gender', 'Race', 'Ethnicity', 'Age',
       'Height (cm)', 'Weight (kg)', 'Indication for Warfarin Treatment',
       'Comorbidities', 'Diabetes',
       'Congestive Heart Failure and/or Cardiomyopathy', 'Valve Replacement',
       'Medications', 'Aspirin', 'Acetaminophen or Paracetamol (Tylenol)',
       'Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day',
       'Simvastatin (Zocor)', 'Atorvastatin (Lipitor)', 'Fluvastatin (Lescol)',
       'Lovastatin (Mevacor)', 'Pravastatin (Pravachol)',
       'Rosuvastatin (Crestor)', 'Cerivastatin (Baycol)',
       'Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)',
       'Phenytoin (Dilantin)', 'Rifampin or Rifampicin',
       'Sulfonamide Antibiotics', 'Macrolide Antibiotics',
       'Anti-fungal Azoles', 'Herbal Medications, Vitamins, Supplements',
       'Target INR', 'Estimated Target INR Range Based on Indication',
       'Subject Reached Stable Dose of Warfarin',
       'Therapeutic Dose of Warfarin',
       'INR on Reported Therapeutic Dose of Warfarin', 'Current Smoker',
       'Cyp2C9 genotypes', 'Genotyped QC Cyp2C9*2', 'Genotyped QC Cyp2C9*3',
       'Combined QC CYP2C9',
       'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
       'VKORC1 QC genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
       'VKORC1 genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C',
       'VKORC1 QC genotype: 497T>G (5808); chr16:31013055; rs2884737; A/C',
       'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G',
       'VKORC1 QC genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G',
       'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G',
       'VKORC1 QC genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G',
       'VKORC1 genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G',
       'VKORC1 QC genotype: 3730 G>A (9041); chr16:31009822; rs7294;  A/G',
       'VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G',
       'VKORC1 QC genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G',
       'VKORC1 genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C',
       'VKORC1 QC genotype: -4451 C>A (861); Chr16:31018002; rs17880887; A/C',
       'CYP2C9 consensus', 'VKORC1 -1639 consensus', 'VKORC1 497 consensus',
       'VKORC1 1173 consensus', 'VKORC1 1542 consensus',
       'VKORC1 3730 consensus', 'VKORC1 2255 consensus',
       'VKORC1 -4451 consensus', 'Unnamed: 63', 'Unnamed: 64', 'Unnamed: 65'],
      dtype='object')
'''