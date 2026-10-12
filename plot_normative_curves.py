"""
Step 3: plots normative curves for given ROIs.

For each sex, plots the healthy normative mean (grey line) ± 1 SD (grey band) as a function
of age, and the new participants' ROI volumes on top of it, one color per group (GROUPS),
with a LOWESS curve per group. With two sessions (SESSIONS), the first is drawn as a dashed
curve and the second as a solid curve with the individual points.

Volumes are recovered from the saved z-scores: volume = z * SD + predicted mean.
The model must expose per-ROI SDs as `stds_` (see config.py). Run fit_normative_model.py first.
"""
import pickle

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

import config

MEAN_HEMISPHERES = False
ROIS_TO_PLOT = config.rois_of_interest(MEAN_HEMISPHERES)  # or a list of ROI names

# Example settings for RLink (lithium response, before and 3 months after lithium).
# GROUP_COLUMN = None plots everyone as one group; SESSIONS = None ignores sessions.
GROUP_COLUMN = "response"
GROUPS = {"PaR/NR": ["PaR", "NR"], "GR": ["GR"]}      # label: values of GROUP_COLUMN 
# RLiNK example: (PaR/NR: partial or no response to treatment; GR: good response to treatment)
SESSIONS = {"M00": "M0 - before lithium", "M03": "M3 - after lithium"}  # first: dashed, second: solid

COLORS = sns.color_palette()


def zscores_to_volumes(model, df_z, roi_idx, roi):
    """Convert z-scores back to ROI volumes."""
    pred = model.predict(df_z[["age", "sex"]])[:, roi_idx]
    return df_z[roi].values * model.stds_[roi_idx] + pred


def plot_participants(ax, df_z, volumes, sex, linestyle, alpha, scatter, lowess_frac):
    """Scatter (optional) + LOWESS curve for each group, for one sex."""
    groups = GROUPS if GROUP_COLUMN else {"Participants": None}
    for (name, values), color in zip(groups.items(), COLORS):
        mask = (df_z["sex"] == sex).values
        if values is not None:
            mask &= df_z[GROUP_COLUMN].isin(values).values
        ages, vols = df_z["age"].values[mask], volumes[mask]
        if scatter and mask.any():
            ax.scatter(ages, vols, s=40, alpha=0.6, color=color, label=f"{name} (n={mask.sum()})")
        if mask.sum() > 3:  # enough points for a meaningful LOWESS
            fit = lowess(vols, ages, frac=lowess_frac)
            ax.plot(fit[:, 0], fit[:, 1], color=color, linewidth=3, alpha=alpha, linestyle=linestyle)


def plot_normative_curve(model, df_z, rois, roi, lowess_frac=0.6):
    """
    Plots ROI volumes (converted back from z-scores) against age on top of the
    normative curve, with one panel per sex.

    Params:
    -----------
    model : fitted normative model (e.g. NormativeBLR), with a per-ROI `stds_` attribute
    df_z : pandas.DataFrame of z-scores, with columns age, sex, the ROIs, and optionally
           session and GROUP_COLUMN
    rois : list of ROI names, in the same order as the model's outputs
    roi : str; name of the ROI to plot
    lowess_frac : float; fraction of data used for LOWESS smoothing
    """
    roi_idx = rois.index(roi)  # same ROI order as the model
    sd = model.stds_[roi_idx]
    volumes = zscores_to_volumes(model, df_z, roi_idx, roi)

    if SESSIONS and "session" in df_z.columns:
        sessions = list(SESSIONS)  # first: dashed curves only, last: solid curves + points
        layers = [(s, "--", 0.4, False) for s in sessions[:-1]] + [(sessions[-1], "-", 0.8, True)]
    else:
        layers = [(None, "-", 0.8, True)]

    age_range = np.linspace(df_z["age"].min(), df_z["age"].max(), 100)
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    for ax, sex in zip(axes, ["female", "male"]):
        # healthy norm
        norm_mean = model.predict(pd.DataFrame({"age": age_range, "sex": sex}))[:, roi_idx]
        ax.plot(age_range, norm_mean, color="grey", linewidth=2, alpha=0.8, label="Healthy mean")
        ax.fill_between(age_range, norm_mean - sd, norm_mean + sd, color="grey", alpha=0.2, label="±1 SD")

        for session, linestyle, alpha, scatter in layers:
            keep = (df_z["session"] == session).values if session else np.ones(len(df_z), bool)
            plot_participants(ax, df_z[keep], volumes[keep], sex, linestyle, alpha, scatter, lowess_frac)

        ax.set_title(sex.capitalize(), fontsize=14, fontweight="bold")
        ax.set_ylabel("Volume (cm³)", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="best")

    axes[1].set_xlabel("Age (years)", fontsize=12)
    fig.suptitle(f"Normative model: {roi}", fontsize=16)
    if len(layers) > 1:
        fig.legend(handles=[mlines.Line2D([], [], color="grey", linestyle=ls, linewidth=2, label=SESSIONS[s])
                            for s, ls, _, _ in layers], loc="upper left", fontsize=12)
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    with open(config.model_path(MEAN_HEMISPHERES), "rb") as f:
        model = pickle.load(f)
    df_z = pd.read_csv(config.zscores_path(MEAN_HEMISPHERES))
    rois = config.roi_columns(df_z)

    for roi in ROIS_TO_PLOT:
        plot_normative_curve(model, df_z, rois, roi)
