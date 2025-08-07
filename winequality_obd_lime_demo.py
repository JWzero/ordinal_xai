"""
Wine Quality – OBD (XGBoost) + LIME demo
Converted from the accompanying Jupyter notebook.
Each logical notebook cell is separated by a `# %%` comment so that it can be
run cell-by-cell in editors like VS Code or Spyder that understand the format.
"""

# %% [markdown] Cell 1 – Title and intro
# Wine Quality Classification with OBD (XGBoost base learner) and LIME Explanations
#
# This script trains an Ordinal Binary Decomposition (OBD) model using an XGBoost
# base learner provided by the `ordinal_xai` package on the UCI Wine Quality
# (red) dataset and explains one test observation twice using Logistic-LIME:
# 1) predictor-effect plot (default) and 2) raw surrogate coefficients.

# %% Cell 2 – Imports & data loading
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# Re-use helper functions from the project’s main pipeline
from main import load_data, load_model, load_interpretation

# %% Cell 3 – Load data using project helper
# The helper expects a dataset name that resides in `ordinal_xai/data/`
DATASET_NAME = "winequality"  # expects `winequality.csv` in the data folder
X, y = load_data(DATASET_NAME)

# Train/test split for evaluation & LIME
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape", X_train.shape, "Test shape", X_test.shape)

# %% Cell 4 – Train OBD (XGBoost base learner)
# %% Cell 4 – Instantiate and train model using project loader
obd_clf = load_model(
    "OBD",
    base_classifier="xgb",
    decomposition_type="one-vs-following",
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
)

print("Fitting OBD model …")
obd_clf.fit(X_train, y_train)

# Evaluate
y_pred = obd_clf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))

# %% Cell 5 – Pick observation index to explain (within training set passed to LIME)
# Use a valid positional index inside X_train after reset_index(drop=True).
idx_to_explain = 1000  # first row
print("Explaining training observation with positional index", idx_to_explain)

# %% Cell 6 – Interpretation using project loader
lime_explainer = load_interpretation(
    "LIME",
    model=obd_clf,
    X=X_train,
    y=y_train,
    model_type="logistic",
    sampling="uniform",
    kernel_width=0.75,
    comparison_method="one_vs_following",
)

# %% Cell 7 – Combined subplot of effects vs. coefficients
# Get LIME results without plotting
result_effect = lime_explainer.explain(observation_idx=idx_to_explain, plot=False, show_coefficients=False)
result_coef   = lime_explainer.explain(observation_idx=idx_to_explain, plot=False, show_coefficients=True)

feature_names = result_effect.get("features")

import numpy as np
fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, max(3, len(feature_names)*0.4)),  # keep width, compress height
    gridspec_kw={"width_ratios": [1, 1]}
)
fig.subplots_adjust(wspace=0.3)  # adjust horizontal spacing
def _fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float, np.number)) else str(v)
obs_vals = ",  ".join([f"{k}: {_fmt(v)}" for k, v in X_train.iloc[idx_to_explain].items()])
fig.suptitle(
    f"LIME Surrogate Comparison (Predictor Effects vs. Coefficients)\nObservation {idx_to_explain} — {obs_vals}",
    fontsize=10
)

# Helper to draw stacked bars (higher & lower) on given axis
def _draw(ax, higher_vals, lower_vals, title):
    n_features = len(feature_names)
    y_high = np.arange(0, n_features * 2, 2)
    y_low = y_high + 1
    ax.barh(y_high, higher_vals, color="#4682b4", label="Higher rank")
    ax.barh(y_low,  lower_vals,  color="#b44646", label="Lower rank")
    yticks = np.concatenate([y_high, y_low])
    ylabels = []
    for n in feature_names:
        ylabels.extend([f"{n} ↑"])
    for n in feature_names:
        ylabels.extend([f"{n} ↓"])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_title(title)
    ax.legend()

# Extract arrays (fallback to coef if effect missing)
high_eff = result_effect.get('higher_effect') if result_effect.get('higher_effect') is not None else result_effect.get('higher_coef')
low_eff  = result_effect.get('lower_effect')  if result_effect.get('lower_effect')  is not None else result_effect.get('lower_coef')
high_coef = result_coef.get('higher_coef')
low_coef  = result_coef.get('lower_coef')

_draw(axes[0], high_eff, low_eff, "Predictor Effects")
_draw(axes[1], high_coef, low_coef, "Raw Surrogate Coefficients")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% Cell 8 – Compare coefficients: CLM vs OBD
# Train a simple CLM model on the same data
clm_model = load_model("CLM")
clm_model.fit(X_train, y_train)

idx_to_explain = 3000

# LIME on CLM
lime_clm = load_interpretation(
    "LIME",
    model=clm_model,
    X=X_train,
    y=y_train,
    model_type="logistic",
    sampling="uniform",
    kernel_width=0.75,
    comparison_method="one_vs_following",
)

result_clm = lime_clm.explain(observation_idx=idx_to_explain, plot=False, show_coefficients=True)
result_obd = lime_explainer.explain(observation_idx=idx_to_explain, plot=False, show_coefficients=True)

# Ensure feature order consistency
feature_names = result_clm.get("features")

fig2, axes2 = plt.subplots(
    1,
    2,
    figsize=(18, max(3, len(feature_names)*0.4)),
    gridspec_kw={"width_ratios": [1, 1]}
)
fig2.subplots_adjust(wspace=0.3)

# CLM coefficients
a_clm_high = result_clm.get('higher_coef')
a_clm_low  = result_clm.get('lower_coef')

# OBD coefficients reuse from earlier
b_obd_high = result_obd.get('higher_coef')
b_obd_low  = result_obd.get('lower_coef')

_draw(axes2[0], a_clm_high, a_clm_low, "CLM Surrogate Coefficients")
_draw(axes2[1], b_obd_high, b_obd_low, "OBD Surrogate Coefficients")

fig2.suptitle(
    f"LIME Coefficient Comparison – Cumulative Link Model vs Ordinal Binary Decomposition Model",
    fontsize=10
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# %% [markdown] Cell 9 – Finished
# You can execute this script directly (`python winequality_obd_lime_demo.py`) or
# run it interactively cell-by-cell in an IDE that recognises the `# %%` syntax.
