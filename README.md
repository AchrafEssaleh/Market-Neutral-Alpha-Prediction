# Market-Neutral Alpha Prediction

## Project Overview
This project addresses a **quantitative finance prediction task**:  
**predicting the sign (up / down) of next-day *market-neutral* stock returns**.

The objective is to build a ML pipeline that extracts predictive signals from historical residual returns and volumes, while remaining neutral to overall market movements.

This repository was developed with a  emphasis on:
- methodological rigor,
- reproducibility,
- clean project structure,
- and interpretable modeling choices.

---

##  Problem Statement
- **Target**: Binary classification  
  → Sign of next-day *residual* return (positive vs negative)
- **Residual return**: Stock return with the market effect removed
- **Metric**: Accuracy on return direction

---

##  Data Description
**Inputs provided by the competition**:
- Past 20 days of residual returns
- Past 20 days of relative trading volumes
- Stock metadata (sector, industry)

 **Raw datasets are intentionally excluded from this repository** due to competition rules.

---

##  Methodology & Pipeline

The project follows a clear & incremental pipeline:

EDA → Baseline Model → Feature Engineering → Advanced Models → Evaluation


### Exploratory Data Analysis (EDA)
- Target distribution analysis (class balance)
- Distributional properties of residual returns
- Volatility structure (heteroskedasticity, skewness)
- Volatility regime analysis (low / medium / high)
- Interaction between momentum and volatility

 **Key takeaway**:  
Raw returns are extremely noisy → predictive power must come from *aggregations*, *regimes*, and *non-linear interactions*.

---

### Baseline Model
- Simple feature set:
  - Lagged returns
  - Lagged volumes
- Baseline classifier (tree-based)
- Purpose:
  - Establish a **reference performance**
  - Validate data pipeline and evaluation logic

---

###  Feature Engineering
- Rolling statistics:
  - Mean returns (5, 10 days)
  - Volatility (5, 10 days)
- Volatility regime encoding
- Momentum-volatility interaction analysis
- Feature importance diagnostics

**Goal**: Extract stable signals from noisy financial time series.

---

### Advanced Models
- Tree-based ensemble models (Random Forest, XGBoost, LightGBM)
- Gradient boosting methods to capture non-linear and regime-dependent effects
- Regularization through depth control, subsampling, and learning rate tuning
- Cross-validation performed at the DATE level to prevent cross-sectional leakage
- Final ensemble model combining multiple learners for robustness

---

###  XX Evaluation XX
- Accuracy comparison vs baseline
- Feature importance analysis
- Stability across volatility regimes
- Final submission file generation

---

##  Repository Structure

Market-Neutral-Alpha-Prediction/
│
├── notebooks/
│ ├── 01_eda_financial.ipynb # Exploratory Data Analysis
│ ├── 02_baseline_model.ipynb # Baseline model
│ ├── 03_feature_engineering.ipynb # Feature engineering
│ ├── 04_models.ipynb # Advanced models
│ ├── X 05_evaluation.ipynb # Evaluation & metrics X
│ ├── figures/ # Plots generated in notebooks
│ └── reports/ # Slides / analysis exports
│
├── src/
│ ├── features/
│ │ └── build_features.py
│ ├── models/
│ │ └── train_model.py
│ └── evaluation/
│ └── evaluate.py
│
├── submissions/
│ └── submission_final.csv # Final competition submission
│
├── data/
│ └── processed/ # Processed datasets (no raw data)
│
├── README.md
├── pyproject.toml
├── uv.lock
├── main.py
└── LICENSE


---

##  Reproducibility
- Python environment managed via `pyproject.toml`
- Deterministic random seeds used where applicable
- Modular pipeline separating exploration and production logic

---

## Data Policy
- Raw CSV files (`data/raw/`) are excluded via `.gitignore`
- Large datasets are not tracked in Git history
- This ensures:
  - compliance with GitHub limits,
  - clean repository,
  - reproducible logic without data leakage

---

##  Results Summary
- Baseline accuracy slightly above random (~50%)
- Feature engineering improves stability and signal extraction
- Tree-based ensembles capture non-linear interactions
- Volatility regimes significantly affect predictability

 **Conclusion**:  
Predicting market-neutral returns is extremely challenging; gains are marginal but systematic feature design and regime-aware modeling provide measurable improvements.

---

## Academic Context
This repository was produced as part of an advanced machine learning  with evaluation criteria focused on:
- sound statistical reasoning,
- code quality,
- experimental clarity,
- and interpretability.

---

##  License
MIT License — see `LICENSE` file.

---

## Contact
For questions or clarifications, feel free to reach out via GitHub.


