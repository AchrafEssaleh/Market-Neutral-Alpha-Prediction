from __future__ import annotations
import numpy as np
import pandas as pd

RET_COLS = [f"RET_{i}" for i in range(1, 21)]
VOL_COLS = [f"VOLUME_{i}" for i in range(1, 21)]
CAT_COLS = ["DATE", "STOCK", "INDUSTRY", "INDUSTRY_GROUP", "SECTOR", "SUB_INDUSTRY"]

def add_features_nn(X: pd.DataFrame, *, keep_original: bool = True) -> pd.DataFrame:
    df = X.copy()

    R = df[RET_COLS].astype(float)
    V = df[VOL_COLS].astype(float)

    # 1) Window stats (mean/std only)
    for w in (3, 5, 10, 20):
        rW = R.iloc[:, :w]
        vW = V.iloc[:, :w]
        df[f"RET_mean_{w}"] = rW.mean(axis=1)
        df[f"RET_std_{w}"]  = rW.std(axis=1)
        df[f"RET_absmean_{w}"] = rW.abs().mean(axis=1)

        df[f"VOL_mean_{w}"] = vW.mean(axis=1)
        df[f"VOL_std_{w}"]  = vW.std(axis=1)

    # 2) Momentum
    df["RET_last1"] = df["RET_1"].astype(float)
    df["RET_last3_sum"]  = R.iloc[:, :3].sum(axis=1)
    df["RET_last5_sum"]  = R.iloc[:, :5].sum(axis=1)
    df["RET_last10_sum"] = R.iloc[:, :10].sum(axis=1)

    # 3) SNR (trend normalized by long vol)
    df["RET_snr_5_20"]  = df["RET_mean_5"]  / (df["RET_std_20"] + 1e-6)
    df["RET_snr_10_20"] = df["RET_mean_10"] / (df["RET_std_20"] + 1e-6)

    # 4) Directionality fractions
    ret_sign = np.sign(R.to_numpy())
    df["RET_pos_frac_20"] = np.nanmean((ret_sign > 0).astype(float), axis=1)
    df["RET_neg_frac_20"] = np.nanmean((ret_sign < 0).astype(float), axis=1)

    # 5) Volume shock / ratio
    df["VOL_last1"] = df["VOLUME_1"].astype(float)
    df["VOL_shock_1_20"] = df["VOL_last1"] - df["VOL_mean_20"]
    df["VOL_ratio_1_20"] = df["VOL_last1"] / (df["VOL_mean_20"] + 1e-6)

    # 6) Interactions
    df["RET1_x_VOL1"] = df["RET_1"].astype(float) * df["VOLUME_1"].astype(float)
    df["ABSRET1_x_VOL1"] = df["RET_1"].astype(float).abs() * df["VOLUME_1"].astype(float)

    # 7) Slopes (trend)
    t = np.arange(1, 21, dtype=np.float32)
    t_mean = t.mean()
    t_var = float(((t - t_mean) ** 2).sum())

    R_np = R.to_numpy()
    V_np = V.to_numpy()
    R_mean = np.nanmean(R_np, axis=1, keepdims=True)
    V_mean = np.nanmean(V_np, axis=1, keepdims=True)

    df["RET_slope_20"] = np.nansum((t - t_mean) * (R_np - R_mean), axis=1) / (t_var + 1e-6)
    df["VOL_slope_20"] = np.nansum((t - t_mean) * (V_np - V_mean), axis=1) / (t_var + 1e-6)

    # 8) Missingness indicators
    df["RET_nan_count"] = np.isnan(R_np).sum(axis=1)
    df["VOL_nan_count"] = np.isnan(V_np).sum(axis=1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if not keep_original:
        keep = []
        if "ID" in df.columns:
            keep.append("ID")
        keep += [c for c in CAT_COLS if c in df.columns]
        engineered = [c for c in df.columns if c not in X.columns]
        return df[keep + engineered]

    return df
