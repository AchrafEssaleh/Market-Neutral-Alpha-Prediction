import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
DATA_PATH = "./data"
OUTPUT_FILE = "submission_baseline.csv"

# --- 2. CHARGEMENT DES DONNÉES ---
print("Chargement des données...")
x_train = pd.read_csv(f'{DATA_PATH}/x_train_Lafd4AH.csv', index_col='ID')
y_train = pd.read_csv(f'{DATA_PATH}/y_train_JQU4vbl.csv', index_col='ID')
test = pd.read_csv(f'{DATA_PATH}/x_test_c7ETL4q.csv', index_col='ID')

train = pd.concat([x_train, y_train], axis=1)
print(f"Données chargées. Train: {train.shape}, Test: {test.shape}")

# --- 3. FEATURE ENGINEERING ---
print("Création des features...")
new_features = []

# Moyennes par secteur et par date
shifts = [1]  
statistics = ['mean']
gb_features = ['SECTOR', 'DATE']
target_feature = 'RET'
tmp_name = '_'.join(gb_features)

for shift in shifts:
    for stat in statistics:
        name = f'{target_feature}_{shift}_{tmp_name}_{stat}'
        feat = f'{target_feature}_{shift}'
        new_features.append(name)
        for data in [train, test]:
            data[name] = data.groupby(gb_features)[feat].transform(stat)

# Définition des colonnes à utiliser
target = 'RET'
n_shifts = 5 
features = ['RET_%d' % (i + 1) for i in range(n_shifts)]
features += ['VOLUME_%d' % (i + 1) for i in range(n_shifts)]
features += new_features

print(f"Features prêtes : {len(features)} variables.")