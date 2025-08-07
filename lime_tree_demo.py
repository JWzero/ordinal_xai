"""Demo: Decision-tree LIME surrogate for the 2-D synthetic ordinal dataset.

Run with
    python lime_tree_demo.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Re-use helpers from the logistic demo
from lime_log_reg_2d_demo import generate_synthetic, train_obd_rf  # noqa: E402

from ordinal_xai.interpretation.lime import LIME  # noqa: E402


def fit_lime_tree(model, X: pd.DataFrame, y: pd.Series, observation_idx: int, kernel_width: Optional[float] = None):
    """Fit a decision-tree LIME surrogate around one observation.

    Returns higher_model, lower_model (each DecisionTreeClassifier or None).
    """
    kw = 0.75 if kernel_width is None else kernel_width
    lime = LIME(model, X, y, model_type="decision_tree", sampling="permute", max_samples=5000, kernel_width=kw)
    result = lime.explain(observation_idx=observation_idx)
    higher_model = result.get("higher_model")
    lower_model = result.get("lower_model")
    return higher_model, lower_model


def plot_tree_boundaries(
    X: pd.DataFrame,
    y: pd.Series,
    obs: pd.Series,
    higher_default,
    lower_default,
    higher_narrow,
    lower_narrow,
):
    """Visualise data & decision-tree LIME boundaries."""
    cmap_pts = ListedColormap(["tab:blue", "tab:orange", "tab:green"])

    # Grid for contours
    xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    grid_df = pd.DataFrame({"x1": xx.ravel(), "x2": yy.ravel()})

    plt.figure(figsize=(8, 6))

    # Default kernel width boundaries (dashed)
    if higher_default is not None:
        proba = higher_default.predict_proba(grid_df)[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, proba, levels=[0.5], colors="tab:green", linestyles="--")
    if lower_default is not None:
        proba = lower_default.predict_proba(grid_df)[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, proba, levels=[0.5], colors="tab:blue", linestyles="--")

    # Narrow kernel width boundaries (dotted)
    if higher_narrow is not None:
        proba = higher_narrow.predict_proba(grid_df)[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, proba, levels=[0.5], colors="tab:green", linestyles=":", linewidths=2)
    if lower_narrow is not None:
        proba = lower_narrow.predict_proba(grid_df)[:, 1].reshape(xx.shape)
        plt.contour(xx, yy, proba, levels=[0.5], colors="tab:blue", linestyles=":", linewidths=2)

    # Scatter points
    plt.scatter(X["x1"], X["x2"], c=y, cmap=cmap_pts, s=60, edgecolor="k", linewidth=0.2, alpha=0.7,
                label="Data points")

    # Highlight observation
    star_handle = plt.scatter(obs["x1"], obs["x2"], c="gold", edgecolor="black", s=240, marker="*",
                label="Explained obs (class=1)")

    # Legend entries for classes
    class_handles = [Patch(facecolor=cmap_pts.colors[i], edgecolor="k", label=f"Class {i}") for i in range(3)]
    boundary_handles = [
        Line2D([0], [0], color="tab:blue", linestyle="--", label="LIME lower – kernel_width=0.75"),
        Line2D([0], [0], color="tab:green", linestyle="--", label="LIME higher – kernel_width=0.75"),
        Line2D([0], [0], color="tab:blue", linestyle=":", linewidth=4, label="LIME lower – kernel_width=0.05"),
        Line2D([0], [0], color="tab:green", linestyle=":", linewidth=4, label="LIME higher – kernel_width=0.05"),
    ]
    plt.legend(handles=class_handles + boundary_handles + [star_handle], title="Legend", loc="upper left")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    # --- fidelity subtitle
    def fidelity(h_model, l_model):
        if h_model is None or l_model is None:
            return float('nan')
        # binary targets
        y_lower = (y == 0).astype(int).values
        y_higher = (y == 2).astype(int).values
        if l_model is not None:
            pred_l = (l_model.predict_proba(X)[:, 1] > 0.5).astype(int)
            acc_lower = (pred_l == y_lower).mean()
        else:
            acc_lower = np.nan
        if h_model is not None:
            pred_h = (h_model.predict_proba(X)[:, 1] > 0.5).astype(int)
            acc_higher = (pred_h == y_higher).mean()
        else:
            acc_higher = np.nan
        return np.nanmean([acc_lower, acc_higher])

    fid_default = fidelity(higher_default, lower_default)
    fid_narrow = fidelity(higher_narrow, lower_narrow)
    plt.title("Decision-tree LIME boundaries on synthetic ordinal data")
    plt.suptitle(f"Fidelity – kernel_width=0.75: {fid_default:.2f}  |  kernel_width=0.05: {fid_narrow:.2f}", fontsize=10, y=0.96)
    plt.tight_layout()
    plt.show()


def main():
    X, y = generate_synthetic()
    model = train_obd_rf(X, y)

    # pick an observation in class 1 roughly centre
    obs_idx = np.where((X["x1"] > 0.73) & (X["x1"] < 0.77) & (X["x2"] > 0.33) & (X["x2"] < 0.37))[0][0]

    # default kernel width (None)
    higher_def, lower_def = fit_lime_tree(model, X, y, obs_idx, kernel_width=None)
    # narrow kernel width
    higher_nar, lower_nar = fit_lime_tree(model, X, y, obs_idx, kernel_width=0.05)

    plot_tree_boundaries(X, y, X.iloc[obs_idx], higher_def, lower_def, higher_nar, lower_nar)


if __name__ == "__main__":
    main()
