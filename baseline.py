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

# --- MODIFICATION ETAPE 5 ---
target = 'RET'
n_shifts = 20  # ON PASSE A 20 JOURS (Maximum disponible)
features = ['RET_%d' % (i + 1) for i in range(n_shifts)]
features += ['VOLUME_%d' % (i + 1) for i in range(n_shifts)]
features += new_features

# AJOUT : Volatilité (Ecart-type des 20 derniers jours)
print("Ajout de la feature Volatilité...")
ret_cols = [f'RET_{i+1}' for i in range(20)]
for data in [train, test]:
    data['VOLATILITY_20'] = data[ret_cols].std(axis=1)

features.append('VOLATILITY_20')

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

#  SOUMISSION 
print("\nEntrainement final sur tout le dataset...")
final_model = RandomForestClassifier(**rf_params)
final_model.fit(X_train_full.fillna(0), y_train_full)

y_test_proba = final_model.predict_proba(test[features].fillna(0))[:, 1]

# Post-processing
sub_test = test.copy()
sub_test['pred_proba'] = y_test_proba
y_test_class = sub_test.groupby('DATE')['pred_proba'].transform(lambda x: x > x.median()).values

submission = pd.Series(y_test_class, index=test.index, name=target)
submission.to_csv(OUTPUT_FILE, index=True, header=True)
print(f"Terminé ! Fichier sauvegardé : {OUTPUT_FILE}")