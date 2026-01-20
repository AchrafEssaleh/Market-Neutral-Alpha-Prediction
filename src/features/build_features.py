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
    Ajoute des features optimisées pour modèles tabulaires type XGBoost / LightGBM :
    - stats fenêtres (mean/std/absmean)
    - momentum
    - ratios/SNR
    - fractions directionnelles
    - volume shock/ratio
    - interactions
    - slopes (trend)
    - missingness
    - range (max-min)
    - ranks intra-date (cross-sectionnels)

    Notes:
    - Les ranks intra-date utilisent uniquement X (pas y) => pas de fuite label.
    - En CV : idéalement recalculer les ranks dans chaque fold (ce module transforme juste X).
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

    # ratios "locaux" (souvent utiles en XGB/LGBM)
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
    # 9) Range (max-min) utile en arbres
    # -----------------------
    df["RET_range_20"] = (R.max(axis=1) - R.min(axis=1)).astype(float)
    df["VOL_range_20"] = (V.max(axis=1) - V.min(axis=1)).astype(float)

    # -----------------------
    # 10) Ranks intra-date (cross-sectional)
    # -----------------------
    if "DATE" in df.columns:
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

def build_dataset_xgb(df_fe: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le dataset final pour XGBoost / LightGBM :
    - conserve ID + colonnes catégorielles + lags bruts RET/VOL
    - conserve toutes les features engineered (fenêtres, ratios, ranks, etc.)
    """
    base_keep: list[str] = []
    if "ID" in df_fe.columns:
        base_keep.append("ID")
    base_keep += [c for c in CAT_COLS if c in df_fe.columns]

    # features engineered = tout ce qui n'est pas ID/CAT/lags bruts
    engineered = [c for c in df_fe.columns if c not in (base_keep + RET_COLS + VOL_COLS)]

    cols = base_keep + [c for c in (RET_COLS + VOL_COLS) if c in df_fe.columns] + engineered
    # dédoublonnage tout en gardant l'ordre
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]

    return df_fe[cols]