# Stock Return Prediction Challenge by QRT

Predict the sign of next day **residual (market neutral)** US stock returns from 20 days of residual returns and relative volumes. Metric: accuracy.

| | Accuracy |
|---|---|
| Benchmark (public leaderboard) | 51.31% |
| **Selected submission (public leaderboard)** | **51.11%** |
| 5 fold CV, grouped by date | 51.74% ± 0.53 |
| Always predicting "down" | 50.11% |

## Key points

- **Fixed a validation leak.** Each date is a full cross section of approx 2,700 stocks, so a random split puts the same day on both sides. Same model: **54.68% with random KFold vs 51.00% with GroupKFold by date.** All results here use the date grouped CV.
- **Leakage safe preprocessing.** Imputation medians (sub industry --> industry --> sector --> global) and the volume cap are learned on train only. Corrupted returns (up to +11,900%) are clipped to ±1.
- **Features:**
  - **v1:** volatility, momentum and short term reversal, built from each stock's own history.
  - **v2:** adds volume (abnormal volume, volume trend, return × volume shock) and same day sector peers (the sector's average return yesterday, stock vs sector).
- **Cross sectional prediction rule.** Each day, stocks are ranked by predicted probability and the top half is predicted "up", which matches a market neutral target.
- **Model choice.** A regularized LightGBM beats the default one by 0.5 points: with this little signal, a flexible model memorizes noise.

## Results (5 fold CV grouped by date)

| Model | v1 | v2 |
|---|---|---|
| Logistic regression | 50.49% | 50.63% |
| Random forest | 51.48% | 51.60% |
| LightGBM, regularized | 51.54% | 51.47% |

| Submission | CV | Public |
|---|---|---|
| A — LightGBM, v1 | 51.62% | 50.80% |
| **B — Random forest, v2** | **51.74%** | **51.11%** |

**What mattered:** the sector peer features, worth +0.31 points on the leaderboard (A --> B). In CV, v2 stays within noise & same day rank features gave no gain. **Honest gap:** B is 0.2 points below the benchmark on the public set. That's within the noise of a small public sample, but not above it.

**Next:** reproduce the benchmark inside the same CV, then add per day rank features on all inputs and sector means for more lags.

## Repository

```
notebooks/eda.ipynb                  cleaning, missing data, outliers
notebooks/feature_engineering.ipynb  v1 and v2 features
notebooks/modeling.ipynb             models, date-grouped CV, submissions
results.csv                          CV log
```

Place `x_train.csv`, `y_train.csv` and `x_test.csv` in `data/raw/` (not tracked), run `uv sync`, then run the notebooks in order.