import numpy as np
import pandas as pd

from xgboost import XGBRanker
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from src.features.build_features import add_features_plus, build_dataset_xgb

# -----------------------
# CONFIG
# -----------------------
RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "RET"
DATA_PATH = "data"
OUTPUT_FILE = "submission_ensemble_soft.csv"


# -----------------------
# UTILS
# -----------------------
def prepare_rank_data(df, features):
    """
    Prépare X, y et groups pour XGBRanker
    """
    X = df[features].fillna(0)
    y = df[TARGET].values
    groups = df.groupby("DATE").size().values
    return X, y, groups


# -----------------------
# CROSS-VALIDATION ENSEMBLE (OPTIMISÉ)
# -----------------------
def cross_val_ensemble(train_df, features):
    print("\nDémarrage de la CV Optimisée (Weighted Voting)...")

    dates = train_df["DATE"].unique()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(dates)):
        d_tr, d_va = dates[tr_idx], dates[va_idx]

        df_tr = train_df[train_df["DATE"].isin(d_tr)].copy()
        df_va = train_df[train_df["DATE"].isin(d_va)].copy()

        # --- 1. XGB RANKER (Le Champion - Paramètres Durcis) ---
        X_tr_xgb, y_tr_xgb, g_tr = prepare_rank_data(df_tr, features)
        X_va_xgb, y_va, g_va = prepare_rank_data(df_va, features)

        xgb = XGBRanker(
            n_estimators=2000,      # AUGMENTATION (avant 1200)
            learning_rate=0.01,     # RALENTISSEMENT (avant 0.03) -> Plus précis
            max_depth=5,            # Légère réduction pour éviter l'overfit
            min_child_weight=100,   # Augmenté pour filtrer le bruit (avant 50)
            subsample=0.6,          # Plus de diversité
            colsample_bytree=0.6,
            reg_alpha=0.5,          # Ajout régularisation L1 (Nettoyage bruit)
            reg_lambda=1.5,         # Ajout régularisation L2
            objective="rank:pairwise",
            eval_metric="ndcg",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        xgb.fit(X_tr_xgb, y_tr_xgb, group=g_tr, verbose=False)
        df_va["score_xgb"] = xgb.predict(X_va_xgb)

        # --- TARGET BINAIRE ---
        y_tr_bin = (df_tr[TARGET] > df_tr.groupby("DATE")[TARGET].transform("median")).astype(int)
        y_va_bin = (df_va[TARGET] > df_va.groupby("DATE")[TARGET].transform("median")).astype(int)

        # --- 2. RANDOM FOREST (La Sécurité) ---
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,            # Réduit de 8 à 6 (RF overfit vite sur ce dataset)
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(df_tr[features].fillna(0), y_tr_bin)
        df_va["score_rf"] = rf.predict_proba(df_va[features].fillna(0))[:, 1]

        # --- 3. LIGHTGBM (Le Challenger) ---
        lgbm = LGBMClassifier(
            n_estimators=1500,      # Augmenté
            learning_rate=0.015,    # Ralenti
            num_leaves=31,          # Plus puissant que max_depth pour LGBM
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        lgbm.fit(df_tr[features].fillna(0), y_tr_bin)
        df_va["score_lgbm"] = lgbm.predict_proba(df_va[features].fillna(0))[:, 1]

        # --- 4. WEIGHTED SOFT VOTING (La Clé du 52%) ---
        # On donne plus de poids au Ranker car il est optimisé pour l'ordre
        # Poids : XGB (50%) + LGBM (30%) + RF (20%)
        
        df_va["score_ensemble"] = (
            0.50 * df_va["score_xgb"].rank(pct=True) +
            0.30 * df_va["score_lgbm"].rank(pct=True) +
            0.20 * df_va["score_rf"].rank(pct=True)
        )

        y_pred = df_va.groupby("DATE")["score_ensemble"].transform(
            lambda x: x > x.median()
        ).astype(int)

        acc = accuracy_score(y_va_bin, y_pred)
        scores.append(acc)

        print(f"Fold {fold + 1} Accuracy: {acc:.4f}")

    final_score = np.mean(scores)

    print("\n" + "=" * 45)
    print(f"FINAL CV ACCURACY (Weighted): {final_score:.4f}")
    print("=" * 45 + "\n")

    return final_score


# -----------------------
# MAIN
# -----------------------
def main():
    print("Chargement des données...")
    train = pd.read_csv(f"{DATA_PATH}/x_train.csv")
    y_train = pd.read_csv(f"{DATA_PATH}/y_train.csv")
    test = pd.read_csv(f"{DATA_PATH}/x_test.csv")

    train = train.merge(y_train, on="ID")

    print("Préparation des features...")
    train_fe = build_dataset_xgb(add_features_plus(train))
    test_fe = build_dataset_xgb(add_features_plus(test))

    FEATURES = [c for c in train_fe.columns if c not in ["RET", "DATE", "ID"]]
    print(f"Nombre de features utilisées: {len(FEATURES)}")

    # --- CV ---
    final_cv_score = cross_val_ensemble(train_fe, FEATURES)
    print(f">>> SCORE FINAL CV: {final_cv_score:.4f}")

    # -----------------------
    # ENTRAINEMENT FINAL
    # -----------------------
    print("\nEntraînement final et prédiction test...")

    # XGB
    X_full, y_full, g_full = prepare_rank_data(train_fe, FEATURES)
    final_xgb = XGBRanker(
        n_estimators=1200,
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
    final_xgb.fit(X_full, y_full, group=g_full, verbose=False)

    # Binaire
    y_bin_full = (train_fe[TARGET] > train_fe.groupby("DATE")[TARGET].transform("median")).astype(int)

    final_rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_rf.fit(train_fe[FEATURES].fillna(0), y_bin_full)

    final_lgbm = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    final_lgbm.fit(train_fe[FEATURES].fillna(0), y_bin_full)

    # -----------------------
    # PREDICTION TEST
    # -----------------------
    X_test = test_fe[FEATURES].fillna(0)

    test_fe["score_xgb"] = final_xgb.predict(X_test)
    test_fe["score_rf"] = final_rf.predict_proba(X_test)[:, 1]
    test_fe["score_lgbm"] = final_lgbm.predict_proba(X_test)[:, 1]

     
    test_fe["score_ensemble"] = (
        0.50 * test_fe["score_xgb"].rank(pct=True) +
        0.30 * test_fe["score_lgbm"].rank(pct=True) +
        0.20 * test_fe["score_rf"].rank(pct=True)
    )

    test_fe["RET"] = test_fe.groupby("DATE")["score_ensemble"].transform(
        lambda x: x > x.median()
    ).astype(bool)

    submission = test_fe[["ID", "RET"]]
    submission.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSubmission saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
