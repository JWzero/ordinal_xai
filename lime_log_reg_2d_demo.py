import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression

# If ordinal_xai is installed as a package, this will work straight away.
# Otherwise, add the project root to the Python path so that we can import it
# when running this script directly from the repository directory.
if __name__ == "__main__":
    import sys
    import os
    project_root = os.path.dirname(__file__)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from ordinal_xai.models import OBD  # noqa: E402
from ordinal_xai.interpretation.lime import LIME  # noqa: E402


def generate_synthetic(n_samples: int = 1000, noise_level: float = 0.1, random_state: int = 42):
    """Generate a 2-D toy dataset with three ordinal classes.

    Parameters
    ----------
    n_samples : int, default=5000
        Number of observations.
    noise_level : float, default=0.1
        Fraction of labels that are randomly flipped to inject noise.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    X : pd.DataFrame of shape (n_samples, 2)
    y : pd.Series of integers in {0,1,2}
    """
    rng = np.random.RandomState(random_state)
    x1 = rng.rand(n_samples)
    x2 = rng.rand(n_samples)

    # Base labels (noise-free)
    y = np.ones(n_samples, dtype=int)  # default class 1
    mask0 = x1 ** 2 + x2 ** 2 > 1
    mask2 = (~mask0) & (x2 < 0.5 * np.sin(4 * x1))
    y[mask0] = 0
    y[mask2] = 2

    # Inject label noise
    n_noisy = int(noise_level * n_samples)
    noisy_idx = rng.choice(n_samples, size=n_noisy, replace=False)
    y[noisy_idx] = rng.randint(0, 3, size=n_noisy)

    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, pd.Series(y, name="y")


def train_obd_rf(X: pd.DataFrame, y: pd.Series):
    """Train an Ordinal Binary Decomposition model with a Random Forest base."""
    model = OBD(base_classifier="rf", n_estimators=200, random_state=42)
    model.fit(X, y)
    return model


def fit_lime_logistic(model, X: pd.DataFrame, y: pd.Series, observation_idx: int = 0):
    """Fit a local LIME surrogate (logistic regression) around a given observation.

    Returns the fitted scikit-learn logistic regression model and the weights
    LIME assigned to each training sample.
    """
    lime = LIME(model, X, y, model_type="logistic", sampling="grid", max_samples=5000)

    # The LIME public API does not expose the surrogate directly, so we re-use
    # the internal machinery to obtain sample weights and then fit a logistic
    # regression ourselves. This is sufficient for visualising a local
    # decision boundary in 2-D.
    result = lime.explain(observation_idx)
    print(result)
    if "higher_coef" in result:
        higher_coef = result["higher_coef"]
    else:
        higher_coef = None
    if "lower_coef" in result:
        lower_coef = result["lower_coef"]
    else:
        lower_coef = None
    if "higher_intercept" in result:
        higher_intercept = result["higher_intercept"]
    else:
        higher_intercept = None
    if "lower_intercept" in result:
        lower_intercept = result["lower_intercept"]
    else:
        lower_intercept = None
    return higher_coef, lower_coef, higher_intercept, lower_intercept


def plot_data_and_boundary(X: pd.DataFrame, y: pd.Series, obs: pd.Series, higher_coef: np.ndarray, lower_coef: np.ndarray, higher_intercept: np.ndarray, lower_intercept: np.ndarray):
    """Visualise the dataset together with the LIME logistic decision boundary."""
    #define color map
    cmap_pts = ListedColormap(["tab:blue", "tab:orange", "tab:green"])
    cmap_bound = ListedColormap(["tab:blue", "tab:orange", "tab:green"])

    xs = np.linspace(0, 1, 200)
    
    #plot higher boundary
    #calculate x2 for higher boundary x1*coef[0]+x2*coef[1]+intercept=0.5
    if higher_coef is not None and higher_intercept is not None:
        x2_higher = (0.5 - higher_intercept - higher_coef[0] * xs) / higher_coef[1]
        plt.plot(xs, x2_higher, color="tab:green", linestyle="--", label="Higher boundary")
    
    if lower_coef is not None and lower_intercept is not None:
        x2_lower = (0.5 - lower_intercept - lower_coef[0] * xs) / lower_coef[1]
        plt.plot(xs, x2_lower, color="tab:blue", linestyle="--", label="Lower boundary")

    
    # Plot data points
    plt.scatter(X["x1"], X["x2"], c=y, cmap=cmap_pts, s=60, edgecolor="k", linewidth=0.2, alpha=0.7)
    
    # Highlight the explained observation and capture its handle
    star_handle = plt.scatter(
        obs["x1"], obs["x2"],
        c="gold", edgecolor="black", s=240, marker="*", label="Explained obs (class=1)"
    )

    # Build legend entries: one patch per class + the star
    class_patches = [
        Patch(facecolor=cmap_pts.colors[i], edgecolor="k", label=f"Class {i}")
        for i in range(3)
    ]

    # Restrict axes to the unit square
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    #add higher and lower boundary to legend as dashed lines    
    boundary_handles = []
    if lower_coef is not None and lower_intercept is not None:
        boundary_handles.append(Line2D([0], [0], color="tab:blue", linestyle="--", label="Lower LIME boundary (class=0)"))
    if higher_coef is not None and higher_intercept is not None:
        boundary_handles.append(Line2D([0], [0], color="tab:green", linestyle="--", label="Higher LIME boundary (class=2)"))
    plt.legend(handles=class_patches + boundary_handles + [star_handle], title="Legend", loc="upper left")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Synthetic ordinal data with LIME logistic surrogate decision boundary")
    plt.tight_layout()
    plt.show()


def main():
    X, y = generate_synthetic()
    obd_model = train_obd_rf(X, y)

    obs_idx = np.where((X["x1"] > 0.73) & (X["x1"] < 0.77) & (X["x2"] > 0.33) & (X["x2"] < 0.37))[0][0]

    # Explain the first observation
    higher_coef, lower_coef, higher_intercept, lower_intercept = fit_lime_logistic(obd_model, X, y, observation_idx=obs_idx)

    print(higher_coef)
    print(lower_coef)

    # Visualise
    plot_data_and_boundary(X, y, X.iloc[obs_idx], higher_coef, lower_coef, higher_intercept, lower_intercept)


if __name__ == "__main__":
    main()
