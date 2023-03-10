from typing import Tuple

import pandas as pd
import numpy as np


RS9923231_COL = "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T"
RS2359612_COL = "VKORC1 genotype: 2255C>T (7566); chr16:31011297; rs2359612; A/G"
RS9934438_COL = "VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G"
RS8050894_COL = "VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G"


def remove_bad_data(df: pd.DataFrame) -> pd.DataFrame:
    reached_stable_dose = df["Subject Reached Stable Dose of Warfarin"] == 1.0
    is_target_known = df["Therapeutic Dose of Warfarin"].notna()
    return df[reached_stable_dose & is_target_known]


def impute_genotypes(df: pd.DataFrame) -> pd.DataFrame:
    is_black = df["Race"] == "Black or African American"
    is_race_unknown = (df["Race"] == "Unknown") | (df["Race"].isna())
    rs2359612_CC = df[RS2359612_COL] == "C/C"
    rs2359612_TT = df[RS2359612_COL] == "T/T"
    rs2359612_CT = df[RS2359612_COL] == "C/T"
    df.loc[rs2359612_CC & ~is_black & ~is_race_unknown, RS9923231_COL] = "G/G"
    df.loc[rs2359612_TT & ~is_black & ~is_race_unknown, RS9923231_COL] = "A/A"
    df.loc[rs2359612_CT & ~is_black & ~is_race_unknown, RS9923231_COL] = "A/G"
    rs9934438_CC = df[RS9934438_COL] == "C/C"
    rs9934438_TT = df[RS9934438_COL] == "T/T"
    rs9934438_CT = df[RS9934438_COL] == "C/T"
    df.loc[rs9934438_CC, RS9923231_COL] = "G/G"
    df.loc[rs9934438_TT, RS9923231_COL] = "A/A"
    df.loc[rs9934438_CT, RS9923231_COL] = "A/G"
    rs8050894_GG = df[RS8050894_COL] == "G/G"
    rs8050894_CC = df[RS8050894_COL] == "C/C"
    rs8050894_CG = df[RS8050894_COL] == "C/G"
    df.loc[rs8050894_GG & ~is_black & ~is_race_unknown, RS9923231_COL] = "G/G"
    df.loc[rs8050894_CC & ~is_black & ~is_race_unknown, RS9923231_COL] = "A/A"
    df.loc[rs8050894_CG & ~is_black & ~is_race_unknown, RS9923231_COL] = "A/G"
    return df


def compute_feature_columns(old_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Age",
        "Height (cm)",
        "Weight (kg)",
        "Race",
        "Carbamazepine (Tegretol)",
        "Phenytoin (Dilantin)",
        "Rifampin or Rifampicin",
        "Amiodarone (Cordarone)",
        "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T",
        "Cyp2C9 genotypes",
        "Therapeutic Dose of Warfarin",
    ]

    df = old_df.copy()
    df = df[df["Therapeutic Dose of Warfarin"].notna() & df["Age"].notna()]
    df = df[columns]

    # Convert race to categories.
    df["race_asian"] = 0.0
    df["race_black"] = 0.0
    df["race_other"] = 0.0
    is_asian = df["Race"] == "Asian"
    is_black = df["Race"] == "Black or African American"
    df.loc[is_asian, "_race_asian"] = 1.0
    df.loc[is_black, "_race_black"] = 1.0
    df.loc[~(is_asian | is_black), "_race_other"] = 1.0

    # Convert VKORC1 genotype to category.
    df["VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T"].fillna(
        "NA", inplace=True
    )
    df["VKORC1 A/G"] = 0.0
    df["VKORC1 A/A"] = 0.0
    df["VKORC1 unknown"] = 0.0
    is_VKORC1_GA = (
        df["VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T"] == "A/G"
    )
    is_VKORC1_AA = (
        df["VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T"] == "A/A"
    )
    is_VKORC1_unknown = (
        df["VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T"] == "NA"
    )
    df.loc[is_VKORC1_GA, "VKORC1 A/G"] = 1.0
    df.loc[is_VKORC1_AA, "VKORC1 A/A"] = 1.0
    df.loc[is_VKORC1_unknown, "VKORC1 unknown"] = 1.0

    # Convert CYP2C9 genotype to category.
    df["Cyp2C9 genotypes"].fillna("NA", inplace=True)
    df["CYP2C9 *1/*2"] = 0.0
    df["CYP2C9 *1/*3"] = 0.0
    df["CYP2C9 *2/*2"] = 0.0
    df["CYP2C9 *2/*3"] = 0.0
    df["CYP2C9 *3/*3"] = 0.0
    df["CYP2C9 unknown"] = 0.0
    is_CYP2C9_12 = df["Cyp2C9 genotypes"] == "*1/*2"
    is_CYP2C9_13 = df["Cyp2C9 genotypes"] == "*1/*3"
    is_CYP2C9_22 = df["Cyp2C9 genotypes"] == "*2/*2"
    is_CYP2C9_23 = df["Cyp2C9 genotypes"] == "*2/*3"
    is_CYP2C9_33 = df["Cyp2C9 genotypes"] == "*3/*3"
    is_CYP2C9_unknown = df["Cyp2C9 genotypes"] == "NA"
    df.loc[is_CYP2C9_12, "CYP2C9 *1/*2"] = 1.0
    df.loc[is_CYP2C9_13, "CYP2C9 *1/*3"] = 1.0
    df.loc[is_CYP2C9_22, "CYP2C9 *2/*2"] = 1.0
    df.loc[is_CYP2C9_23, "CYP2C9 *2/*3"] = 1.0
    df.loc[is_CYP2C9_33, "CYP2C9 *3/*3"] = 1.0
    df.loc[is_CYP2C9_unknown, "CYP2C9 unknown"] = 1.0

    df["Height (cm)"].fillna(
        df["Height (cm)"].mean(numeric_only=True).round(1), inplace=True
    )
    df["Weight (kg)"].fillna(
        df["Weight (kg)"].mean(numeric_only=True).round(1), inplace=True
    )
    df["Age"].fillna("40 - 49", inplace=True)
    # Convert age to decade.
    df["_age_decade"] = 0.0
    df = df.assign(_age_decade=lambda x: int(x["Age"][0][0]))
    df = df.drop(
        columns=[
            "Age",
            "Race",
            "VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T",
            "Cyp2C9 genotypes",
        ]
    )
    df = df.fillna(0.0)
    return df


def preprocess_patients_df(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_bad_data(df)
    df = impute_genotypes(df)
    df = compute_feature_columns(df)
    return df


if __name__ == "__main__":
    PATIENTS_FILE = "data/warfarin.csv"
    patients_df = pd.read_csv("data/warfarin.csv")

    patients_df = remove_bad_data(patients_df)
    print(patients_df[RS9923231_COL].count())
    patients_df = impute_genotypes(patients_df)
    print(patients_df[RS9923231_COL].count())


def compute_features_and_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset ordering

    n_samples, n_arms = len(df), 3

    features_df = df.drop("Therapeutic Dose of Warfarin", axis=1)
    features_np = np.hstack([features_df.to_numpy(), np.ones((n_samples, 1))])

    dosage_np = df["Therapeutic Dose of Warfarin"].to_numpy()
    targets_np = np.zeros((n_samples, n_arms))
    targets_np[dosage_np < 21, 1] = -1.0
    targets_np[dosage_np < 21, 2] = -1.0
    targets_np[np.all([dosage_np >= 21, dosage_np <= 49], axis=0), 0] = -1.0
    targets_np[np.all([dosage_np >= 21, dosage_np <= 49], axis=0), 2] = -1.0
    targets_np[dosage_np > 49, 0] = -1.0
    targets_np[dosage_np > 49, 1] = -1.0
    return features_np, targets_np


"""
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
"""
