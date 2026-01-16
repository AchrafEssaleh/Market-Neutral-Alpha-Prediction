# feature_engineering_plus.py
from __future__ import annotations

import numpy as np
import pandas as pd

RET_COLS = [f"RET_{i}" for i in range(1, 21)]
VOL_COLS = [f"VOLUME_{i}" for i in range(1, 21)]
CAT_COLS = ["DATE", "STOCK", "INDUSTRY", "INDUSTRY_GROUP", "SECTOR", "SUB_INDUSTRY"]

# --- Utilitaires ---
def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
    return a / (b + eps)

def add_features_plus(X: pd.DataFrame, *, keep_original: bool = True) -> pd.DataFrame:
    """
    Ajoute des features NN-friendly + des features "tabular boost" utiles pour XGBoost :
    - ratios locaux (trend/vol) supplémentaires
    - range (max-min)
    - ranks intra-date (cross-sectionnels) sur quelques signaux clés

    IMPORTANT:
    - Les ranks intra-date utilisent uniquement X (pas y) donc pas de fuite "label".
    - En CV, idéalement recalculer ces ranks dans chaque fold (mais ce module ne fait que transformer X).
    """
    df = X.copy()

    # --- Numeric matrices (NaN ok) ---
    R = df[RET_COLS].astype(float)
    V = df[VOL_COLS].astype(float)

    # -----------------------
    # 1) Window stats (mean/std/absmean)
    # -----------------------
    for w in (3, 5, 10, 20):
        rW = R.iloc[:, :w]
        vW = V.iloc[:, :w]

        df[f"RET_mean_{w}"] = rW.mean(axis=1)
        df[f"RET_std_{w}"] = rW.std(axis=1)
        df[f"RET_absmean_{w}"] = rW.abs().mean(axis=1)

        df[f"VOL_mean_{w}"] = vW.mean(axis=1)
        df[f"VOL_std_{w}"] = vW.std(axis=1)

    # -----------------------
    # 2) Momentum
    # -----------------------
    df["RET_last1"] = df["RET_1"].astype(float)
    df["RET_last3_sum"] = R.iloc[:, :3].sum(axis=1)
    df["RET_last5_sum"] = R.iloc[:, :5].sum(axis=1)
    df["RET_last10_sum"] = R.iloc[:, :10].sum(axis=1)

    # -----------------------
    # 3) SNR (global + ratios locaux)
    # -----------------------
    df["RET_snr_5_20"] = _safe_div(df["RET_mean_5"], df["RET_std_20"])
    df["RET_snr_10_20"] = _safe_div(df["RET_mean_10"], df["RET_std_20"])

    # 🔥 NOUVEAU : ratios "locaux" (souvent utiles en XGB)
    df["RET_snr_3_5"] = _safe_div(df["RET_mean_3"], df["RET_std_5"])
    df["RET_snr_5_5"] = _safe_div(df["RET_mean_5"], df["RET_std_5"])
    df["RET_snr_10_10"] = _safe_div(df["RET_mean_10"], df["RET_std_10"])

    # -----------------------
    # 4) Directionality fractions
    # -----------------------
    ret_sign = np.sign(R.to_numpy())  # -1,0,1, NaN -> NaN
    df["RET_pos_frac_20"] = np.nanmean((ret_sign > 0).astype(float), axis=1)
    df["RET_neg_frac_20"] = np.nanmean((ret_sign < 0).astype(float), axis=1)

    # -----------------------
    # 5) Volume shock / ratio
    # -----------------------
    df["VOL_last1"] = df["VOLUME_1"].astype(float)
    df["VOL_shock_1_20"] = df["VOL_last1"] - df["VOL_mean_20"]
    df["VOL_ratio_1_20"] = _safe_div(df["VOL_last1"], df["VOL_mean_20"])

    # -----------------------
    # 6) Interactions
    # -----------------------
    df["RET1_x_VOL1"] = df["RET_1"].astype(float) * df["VOLUME_1"].astype(float)
    df["ABSRET1_x_VOL1"] = df["RET_1"].astype(float).abs() * df["VOLUME_1"].astype(float)

    # -----------------------
    # 7) Slopes (trend)
    # -----------------------
    t = np.arange(1, 21, dtype=np.float32)
    t_mean = t.mean()
    t_var = float(((t - t_mean) ** 2).sum())

    R_np = R.to_numpy()
    V_np = V.to_numpy()
    R_mean = np.nanmean(R_np, axis=1, keepdims=True)
    V_mean = np.nanmean(V_np, axis=1, keepdims=True)

    df["RET_slope_20"] = np.nansum((t - t_mean) * (R_np - R_mean), axis=1) / (t_var + 1e-6)
    df["VOL_slope_20"] = np.nansum((t - t_mean) * (V_np - V_mean), axis=1) / (t_var + 1e-6)

    # -----------------------
    # 8) Missingness indicators
    # -----------------------
    df["RET_nan_count"] = np.isnan(R_np).sum(axis=1)
    df["VOL_nan_count"] = np.isnan(V_np).sum(axis=1)

    # -----------------------
    # 9) 🔥 NOUVEAU : range (max-min) utile en arbres
    # -----------------------
    # Ici on le calcule uniquement sur 20 (stabilité)
    df["RET_range_20"] = (R.max(axis=1) - R.min(axis=1)).astype(float)
    df["VOL_range_20"] = (V.max(axis=1) - V.min(axis=1)).astype(float)

    # -----------------------
    # 10) 🔥 NOUVEAU : ranks intra-date (cross-sectional)
    # -----------------------
    # Ces ranks comparent une action aux autres le même jour.
    # Très utiles pour les modèles tabulaires (XGB/LightGBM), parfois aussi pour MLP.
    if "DATE" in df.columns:
        # on rank en percentile (0..1)
        df["RET_last1_rank_date"] = df.groupby("DATE")["RET_last1"].rank(pct=True, method="average")
        df["VOL_last1_rank_date"] = df.groupby("DATE")["VOL_last1"].rank(pct=True, method="average")
        df["RET_mean_5_rank_date"] = df.groupby("DATE")["RET_mean_5"].rank(pct=True, method="average")
        df["VOL_ratio_1_20_rank_date"] = df.groupby("DATE")["VOL_ratio_1_20"].rank(pct=True, method="average")

    # Clean infs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if not keep_original:
        keep = []
        if "ID" in df.columns:
            keep.append("ID")
        keep += [c for c in CAT_COLS if c in df.columns]
        engineered = [c for c in df.columns if c not in X.columns]
        return df[keep + engineered]

    return df

def build_dataset_for_model(df_fe: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    df_fe = DataFrame après add_features_plus(...)

    model:
      - "xgb" : on garde quasi tout (arbres aiment redondance + ranks + ratios)
      - "mlp" : on garde lags bruts + un set compact (pas trop redondant)
      - "seq" : on garde surtout la séquence (RET/VOL) + qq global features + cats

    Retourne un df prêt à être séparé en (num, cat) ou en séquence.
    """
    model = model.lower().strip()
    base_keep = []
    if "ID" in df_fe.columns:
        base_keep.append("ID")
    base_keep += [c for c in CAT_COLS if c in df_fe.columns]

    # Toutes les engineered cols (ce qui n'est pas base lags/cats/ID)
    # On va plutôt construire explicitement ce qu'on veut garder.
    engineered_all = [c for c in df_fe.columns if c not in (base_keep + RET_COLS + VOL_COLS)]

    # --- familles ---
    window_stats = [c for c in df_fe.columns if (
        c.startswith("RET_mean_") or c.startswith("RET_std_") or c.startswith("RET_absmean_") or
        c.startswith("VOL_mean_") or c.startswith("VOL_std_")
    )]
    momentum = [c for c in ["RET_last1", "RET_last3_sum", "RET_last5_sum", "RET_last10_sum"] if c in df_fe.columns]
    snr = [c for c in df_fe.columns if c.startswith("RET_snr_")]
    direction = [c for c in ["RET_pos_frac_20", "RET_neg_frac_20"] if c in df_fe.columns]
    volume_shock = [c for c in ["VOL_last1", "VOL_shock_1_20", "VOL_ratio_1_20"] if c in df_fe.columns]
    interactions = [c for c in ["RET1_x_VOL1", "ABSRET1_x_VOL1"] if c in df_fe.columns]
    slopes = [c for c in ["RET_slope_20", "VOL_slope_20"] if c in df_fe.columns]
    missing = [c for c in ["RET_nan_count", "VOL_nan_count"] if c in df_fe.columns]
    ranges = [c for c in ["RET_range_20", "VOL_range_20"] if c in df_fe.columns]
    ranks_date = [c for c in df_fe.columns if c.endswith("_rank_date")]

    if model == "xgb":
        # XGBoost aime:
        # - lags bruts
        # - stats fenêtres
        # - ratios / ranks
        # - redondance OK
        keep_num = RET_COLS + VOL_COLS + window_stats + momentum + snr + direction + volume_shock + interactions + slopes + missing + ranges + ranks_date
        cols = base_keep + [c for c in keep_num if c in df_fe.columns]
        return df_fe[cols]

    if model == "mlp":
        # MLP+embeddings:
        # - on garde les lags bruts (séquence "aplatie")
        # - et un set compact (éviter trop de redondance)
        # => on garde surtout mean/std sur (5,10,20) + quelques signaux globaux + missing + 1-2 ranks
        compact_window = [c for c in window_stats if any(c.endswith(f"_{w}") for w in (5, 10, 20))]
        compact_snr = [c for c in snr if c in ("RET_snr_5_20", "RET_snr_10_20", "RET_snr_10_10", "RET_snr_3_5")]
        compact_ranks = [c for c in ranks_date if c in ("RET_last1_rank_date", "VOL_last1_rank_date", "RET_mean_5_rank_date")]

        keep_num = RET_COLS + VOL_COLS + compact_window + momentum + compact_snr + direction + volume_shock + interactions + slopes + missing + compact_ranks
        cols = base_keep + [c for c in keep_num if c in df_fe.columns]
        return df_fe[cols]

    if model == "seq":
        # Modèle séquentiel (CNN/LSTM):
        # - priorité à la séquence brute RET/VOL
        # - + quelques features globales utiles (missing + ratio volume)
        # - ranks optionnels (mais pas indispensables)
        keep_num = RET_COLS + VOL_COLS + missing + ["VOL_ratio_1_20"]
        keep_num += [c for c in ["RET1_x_VOL1"] if c in df_fe.columns]  # optionnel

        cols = base_keep + [c for c in keep_num if c in df_fe.columns]
        return df_fe[cols]

    raise ValueError("model must be one of: 'xgb', 'mlp', 'seq'")