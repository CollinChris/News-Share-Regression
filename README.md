<<<<<<< HEAD
# News share prediction Regression Pipeline
This machine learning pipeline takes in a dataset of news article data and uses regression models to predict shares.

> **Goal**: Predict `shares` (highly skewed) using article metadata, content features, and publishing context with robust preprocessing, feature engineering, and hyperparameter-tuned regression models.

---

## Key Results (Final Test Set)
| Model               | R²   | MAE (log) | MAE (original) | RMSE |
|---------------------|------|-----------|----------------|------|
| **XGBoost (Tuned)** | **0.46** | **0.47** | **1,454.76** | 0.67 |
| Random Forest       | 0.42 | 0.50      | 1,494.47       | 0.70 |
| Linear Regression   | 0.26 | 0.60      | 1,778.15       | 0.78 |

> **XGBoost** is the final selected model — explains **46%** of variance in log-shares and predicts within **~1,455 shares** on average.

---

## Prerequisites & Installation

| Step | Command |
|------|--------|
| Install dependencies | `pip install -r requirements.txt` |

> Uses `scikit-learn`, `xgboost`, `pandas`, `numpy`, `yaml`, `joblib`, `seaborn`, `matplotlib`.

---

## Executing Pipeline

| Method | Command |
|--------|--------|
| Run script | `bash run.sh` |
| Make executable & run | `chmod +x run.sh && ./run.sh` |

> Logs saved to `training_log.txt`. All configs in `src/config.yaml`.

---

## Configuration (`src/config.yaml`)

| Parameter | Purpose |
|---------|--------|
| `target` | Target column (`shares`) |
| `numerical_features` | List of numeric columns |
| `categorical_features` | List of categorical columns |
| `param_grid` | Hyperparameter search space |
| `cv` | Cross-validation folds |
| `scoring` | Evaluation metric (`r2`) |

> All parameters are **externalized** for easy tuning without code changes.

---

## Data Dictionary (`news.csv`)

| Feature | Description |
|--------|-------------|
| `ID`, `URL` | Article identifiers |
| `timedelta` | Days since publication |
| `weekday`, `data_channel` | Publishing context |
| `n_tokens_title`, `n_tokens_content` | Title & content length |
| `n_unique_tokens`, `average_token_length` | Readability |
| `num_hrefs`, `num_imgs`, `num_videos` | Multimedia |
| `n_comments` | Reader engagement |
| `self_reference_*_shares` | Internal linking |
| `num_keywords`, `kw_*` | Keyword performance |
| **`shares`** | **Target**: Number of social shares |

---

## Pipeline Flow

| Step | Action |
|------|--------|
| 1. EDA & Cleaning | Inspect patterns, fix negatives, cap outliers |
| 2. Feature Engineering | Log-transform target, create `n_comments × data_channel` |
| 3. Train/Val/Test Split | 60% / 20% / 20% |
| 4. Preprocessing | `log1p` + `RobustScaler`, `OneHotEncoder` |
| 5. Baseline Models | Train Linear, RF, XGBoost |
| 6. Hyperparameter Tuning | `GridSearchCV` (5-fold) |
| 7. Refit Best Model | On **train + val** |
| 8. Final Test Evaluation | Unbiased metrics on held-out test set |

---

## Overview of Submitted folder
```text
.
├── eda.ipnyb
├── data
│   └── news.csv
├── requirements.txt
├── training log.txt
├── run.sh
├── main.py
├── eda.ipynb
└── src
    └── data_preparation.py
    └── model_training.py
    └── config.yaml
    └── eda.py               # A class with functions to assist EDA 
    └── missingness.py       # A class with functions to assess missing data
```

## EDA Findings
| Insight | Evidence |
|---------|----------|
| **Extreme right-skew in `shares`** | Mean: ~3,379, Median: 1,400, Max: 663,600 → **99th percentile = 32,000**. Log-transform (`log1p`) applied to target. |
| **`n_comments` is the strongest predictor** | Mutual Information: **1.66**, Spearman correlation: **0.36** → top feature in both linear and tree models. |
| **Data channel influences engagement** | `data_channel='tech'` and `'entertainment'` have higher median shares. Interaction features (`n_comments × data_channel`) improved R² by **~0.03**. |
| **Keyword performance features are noisy** | `kw_min_min = -1` in **20,660 rows** (impossible), `kw_avg_min` negative in **767** → replaced with 0. High variance, low predictive power. |
| **Multimedia content has weak signal** | `num_imgs`, `num_videos` → near-zero correlation with `shares` after log-transform. Not selected in final models. |
| **Outliers dominate raw metrics** | 4,080 articles exceed IQR bounds for `shares`. Capped at **99th percentile (32,000)** and `n_comments` at **1,188**. |
| **Temporal patterns exist** | Articles published on **weekends** and **Mondays** show slightly higher shares. `weekday` encoded via one-hot. |
| **Self-references matter moderately** | `self_reference_avg_shares` → MI = 0.41, but noisy due to many zeros. Retained in final feature set. |

> **Full interactive analysis**: `notebooks/eda.ipynb`  
> **Visualizations**: Distribution plots, correlation heatmaps, interaction scatterplots, feature importance (MI), outlier boxplots.
> **Key Takeaway**: The data is **highly skewed and noisy**, but **reader engagement (`n_comments`)** and **publishing context (`data_channel`, `weekday`)** are the most reliable signals for predicting virality.

## Feature Handling

| Action | Rationale |
|--------|-----------|
| **Impute negatives & missing values** | `kw_min_min`, `num_videos`, etc. → filled with `0` or `'unknown'` (assumed parsing error). |
| **Add `is_missing_videos` flag** | Marks articles with missing multimedia data for model awareness. |
| **Log-transform numerical features** | Handles extreme skew in `n_comments`, `num_hrefs`, `kw_avg_avg`, etc. |
| **Cap top 1% outliers** | `shares` ≤ **32,000**, `n_comments` ≤ **1,188** → prevents rare viral cases from dominating training. |
| **Drop low-correlation features** | Post-baseline testing, removed noisy columns (e.g., `kw_min_max`) to reduce overfitting. |
| **Engineer interaction features** | `n_comments × data_channel` → captures channel-specific engagement (e.g., tech readers comment more). |

---

## Models Used

| Model | Role & Rationale |
|-------|------------------|
| **1. Linear Regression** | Baseline model for interpretability and quick benchmarking. Assumes linear relationships — useful to establish a performance floor (R² = 0.26 on validation). |
| **2. SVM Regression** | *Explored but excluded.* SVR struggled with **35k+ samples** and high-dimensional features post-encoding. Scaling was slow, and performance lagged behind tree models (R² < 0.30). |
| **3. Random Forest Regression** | Ensemble of decision trees — captures **non-linear patterns** and **feature interactions** naturally. Robust to outliers. Achieved **R² = 0.42** on validation. Strong baseline before boosting. |
| **4. XGBoost Regression** | **Final selected model.** Gradient-boosted trees with regularization. Best performance: **R² = 0.46**, **MAE (original) = 1,454.76** on test set. Handles skewness, interactions, and noisy features effectively. |

> **Why XGBoost Won**: Outperformed Random Forest by **+0.04 R²** and reduced original-scale error by **~40 shares** on average. Efficient with `n_jobs=-1`, supports early stopping, and provides built-in feature importance.

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|--------|----------------|
| **MAE (log scale)** | `|y - ŷ|` | Average prediction error in log-shares: **~0.47** |
| **MAE (original scale)** | `|expm1(y) - expm1(ŷ)|` | Real-world error: **~1,455 shares** |
| **RMSE** | `√MSE` | Sensitive to large errors; **0.67** in log space |
| **R²** | `1 - SS_res / SS_tot` | Proportion of variance explained: **46%** |

> All metrics computed on **held-out test set (20%)** — no data leakage.

---

## Considerations for Deployment

| Aspect | Implementation |
|--------|----------------|
| **Reproducibility** | Fixed `requirements.txt`, `config.yaml`, `random_state=42` |
| **Configuration Flexibility** | All features, models, and tuning in `config.yaml` |
| **Scalability** | Save model with `joblib.dump()` → load in Flask/FastAPI |
| **Monitoring & Retraining** | Track `shares` distribution drift; retrain quarterly |
| **Explainability** | Use SHAP values from XGBoost for feature importance |

---

## Conclusion

| Statement | Justification |
|---------|---------------|
| **R² = 0.46 is strong for news virality** | `shares` is **inherently noisy** — driven by external events, timing, and luck. Most articles get **< 2,000 shares**; viral outliers (100k+) are rare and unpredictable. |
| **MAE = 1,455 is practically useful** | For a median article (~1,400 shares), predicting within **±1,455** is **> 50% relative accuracy** on typical cases. |
| **Log-transform + outlier capping was essential** | Without it, models overfit to rare viral hits → poor generalization. Final model learns **representative patterns**. |
| **XGBoost + interactions = robust solution** | Captures non-linear engagement (e.g., `n_comments` in `tech` channel) without overfitting. |

> **Final Verdict**: **R² = 0.46 is acceptable** for real-world news share prediction. The model reliably flags **high-potential articles** for editorial promotion with **data-driven confidence**.
=======
# CollinChris-RegressionMiniProject
>>>>>>> 7c89254892d5ea93858af709197880659ab16648
