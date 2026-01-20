# src/models/train_ensemble_lgbm.py

import numpy as np
import pandas as pd
from xgboost import XGBRanker
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from src.features.build_features import add_features_plus, build_dataset_xgb

RANDOM_STATE = 42
TARGET = "RET"
DATA_PATH = "data"
OUTPUT_FILE = "submission_ensemble_lgbm.csv"

# --- Préparation des données ---
def prepare_rank_data(df, features):
    X = df[features].fillna(0)
    y = df[TARGET].values
    groups = df.groupby("DATE").size().values
    return X, y, groups

# --- XGBRanker cross-validation ---
def cross_val_xgb(train_df, features, n_splits=5):
    dates = train_df["DATE"].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(dates)):
        d_tr, d_va = dates[tr_idx], dates[va_idx]
        df_tr = train_df[train_df["DATE"].isin(d_tr)]
        df_va = train_df[train_df["DATE"].isin(d_va)]

        X_tr, y_tr, g_tr = prepare_rank_data(df_tr, features)
        X_va, y_va, g_va = prepare_rank_data(df_va, features)

        model = XGBRanker(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=50,
            subsample=0.7,
            colsample_bytree=0.7,
            objective="rank:pairwise",
            eval_metric="ndcg",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_va, y_va)], eval_group=[g_va], verbose=False)

        df_va = df_va.copy()
        df_va["score"] = model.predict(X_va)
        y_pred = df_va.groupby("DATE")["score"].transform(lambda x: x > x.median()).astype(int)

        acc = accuracy_score(y_va, y_pred)
        scores.append(acc)

    return np.mean(scores)

# --- Random Forest cross-validation ---
def cross_val_rf(train_df, features, n_splits=5):
    dates = train_df["DATE"].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(dates)):
        d_tr, d_va = dates[tr_idx], dates[va_idx]
        df_tr = train_df[train_df["DATE"].isin(d_tr)]
        df_va = train_df[train_df["DATE"].isin(d_va)]

        X_tr = df_tr[features].fillna(0)
        y_tr = (df_tr[TARGET] > 0).astype(int)
        X_va = df_va[features].fillna(0)
        y_va = (df_va[TARGET] > 0).astype(int)

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)

        df_va = df_va.copy()
        df_va["score"] = model.predict_proba(X_va)[:, 1]
        y_pred = df_va.groupby("DATE")["score"].transform(lambda x: x > x.median()).astype(int)
        acc = accuracy_score(y_va, y_pred)
        scores.append(acc)

    return np.mean(scores)

# --- LightGBM cross-validation ---
def cross_val_lgbm(train_df, features, n_splits=5):
    dates = train_df["DATE"].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(dates)):
        d_tr, d_va = dates[tr_idx], dates[va_idx]
        df_tr = train_df[train_df["DATE"].isin(d_tr)]
        df_va = train_df[train_df["DATE"].isin(d_va)]

        X_tr = df_tr[features].fillna(0)
        y_tr = (df_tr[TARGET] > 0).astype(int)
        X_va = df_va[features].fillna(0)
        y_va = (df_va[TARGET] > 0).astype(int)

        model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        df_va = df_va.copy()
        df_va["score"] = model.predict_proba(X_va)[:, 1]
        y_pred = df_va.groupby("DATE")["score"].transform(lambda x: x > x.median()).astype(int)
        acc = accuracy_score(y_va, y_pred)
        scores.append(acc)

    return np.mean(scores)

# --- Main ---
def main():
    print("Loading data...")
    train = pd.read_csv(f"{DATA_PATH}/x_train.csv")
    test = pd.read_csv(f"{DATA_PATH}/x_test.csv")

    print("Feature engineering...")
    train_fe = build_dataset_xgb(add_features_plus(train))
    test_fe = build_dataset_xgb(add_features_plus(test))
    FEATURES = [c for c in train_fe.columns if c not in ["RET", "DATE", "ID"]]

    # --- Cross-validation des trois modèles ---
    score_xgb = cross_val_xgb(train_fe, FEATURES)
    score_rf = cross_val_rf(train_fe, FEATURES)
    score_lgbm = cross_val_lgbm(train_fe, FEATURES)

    print(f"\nCV Scores:\nXGBRanker: {score_xgb:.4f}\nRandom Forest: {score_rf:.4f}\nLightGBM: {score_lgbm:.4f}")

    # --- Entrainement final ---
    X_train, y_train, g_train = prepare_rank_data(train_fe, FEATURES)
    final_xgb = XGBRanker(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=50,
        subsample=0.7,
        colsample_bytree=0.7,
        objective="rank:pairwise",
        eval_metric="ndcg",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_xgb.fit(X_train, y_train, group=g_train, verbose=False)

    final_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_rf.fit(X_train, (y_train > 0).astype(int))

    final_lgbm = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_lgbm.fit(X_train, (y_train > 0).astype(int))

    # --- Prédiction test ---
    X_test, _, _ = prepare_rank_data(test_fe, FEATURES)
    test_fe = test_fe.copy()
    test_fe["score_xgb"] = final_xgb.predict(X_test)
    test_fe["score_rf"] = final_rf.predict_proba(X_test)[:, 1]
    test_fe["score_lgbm"] = final_lgbm.predict_proba(X_test)[:, 1]

    # --- Ensemble simple (moyenne des 3 modèles) ---
    test_fe["score_ensemble"] = (
        test_fe["score_xgb"] + test_fe["score_rf"] + test_fe["score_lgbm"]
    ) / 3.0

    y_test_bool = test_fe.groupby("DATE")["score_ensemble"].transform(
        lambda x: x > x.median()
    )
    y_test_bool = y_test_bool.astype(bool)  # True si > median, False sinon

    submission = pd.DataFrame({
        "ID": test_fe.index,
        "RET": y_test_bool
    })
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

