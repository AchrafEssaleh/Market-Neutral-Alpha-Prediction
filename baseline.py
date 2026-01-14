import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

#  CONFIGURATION
DATA_PATH = "./data"
OUTPUT_FILE = "submission_baseline.csv"

#  CHARGEMENT DES DONNÉES 
print("Chargement des données...")
x_train = pd.read_csv(f'{DATA_PATH}/x_train_Lafd4AH.csv', index_col='ID')
y_train = pd.read_csv(f'{DATA_PATH}/y_train_JQU4vbl.csv', index_col='ID')
test = pd.read_csv(f'{DATA_PATH}/x_test_c7ETL4q.csv', index_col='ID')

train = pd.concat([x_train, y_train], axis=1)
print(f"Données chargées. Train: {train.shape}, Test: {test.shape}")

# FEATURE ENGINEERING 
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
# Cross Validation
X_train_full = train[features]
y_train_full = train[target]

rf_params = {
    'n_estimators': 500,
    'max_depth': 8,
    'random_state': 0,
    'n_jobs': -1
}

train_dates = train['DATE'].unique()
kf = KFold(n_splits=4, random_state=0, shuffle=True)

scores = []
print("\nLancement de la validation croisée...")

for i, (train_idx, val_idx) in enumerate(kf.split(train_dates)):
    date_train, date_val = train_dates[train_idx], train_dates[val_idx]
    
    mask_train = train['DATE'].isin(date_train)
    mask_val = train['DATE'].isin(date_val)
    
    # Remplissage des NaN par 0
    X_loc_train = X_train_full[mask_train].fillna(0)
    y_loc_train = y_train_full[mask_train]
    X_loc_val = X_train_full[mask_val].fillna(0)
    y_loc_val = y_train_full[mask_val]

    model = RandomForestClassifier(**rf_params)
    model.fit(X_loc_train, y_loc_train)
    
    # Prédiction et transformation en médiane par jour
    y_pred_proba = model.predict_proba(X_loc_val)[:, 1]
    sub_val = train.loc[mask_val].copy()
    sub_val['pred_proba'] = y_pred_proba
    y_pred_class = sub_val.groupby('DATE')['pred_proba'].transform(lambda x: x > x.median()).values
    
    score = accuracy_score(y_loc_val, y_pred_class)
    scores.append(score)
    print(f"Fold {i+1}: {score*100:.2f}%")

print(f"Score Moyen Validation: {np.mean(scores)*100:.2f}%")