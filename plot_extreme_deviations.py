"""
Step 4: brain maps displaying the % of participants with strong deviations from the norm (above and below separately), 
for all brain regions.

Works with any set of new participants whose z-scores were computed by fit_normative_model.py
(config.zscores_path). Set the dataset-specific options below: which session to use (if the
data has several), and which column defines the groups (if any).

For each group, computes per ROI the percentage of participants with an extreme z-score below
the norm (z < -Z_THRESHOLD) and above the norm (z > +Z_THRESHOLD). The two directions are
counted separately, so that a region where some participants are far below the norm and others
far above isn't mistaken for a consistent deviation.

Under the norm, about 2.5% of participants are expected beyond ±1.96 in each direction (Gaussian distrib used as norm). 
Each percentage is compared to that expected rate with a one-sided binomial test, corrected for the
number of ROIs (FDR). Glass brains show the percentages of the ROIs that remain significant,
one map below the norm and one above, for one tissue (TISSUE_SUFFIX) at a time, with the same
color scale for every group. 

An ROI significant in both directions is heterogeneous (some participants far below the norm, others far above) 
rather than consistently shifted; the script lists those ROIs. 
The results of all groups are also saved as a csv next to the z-scores.

Note:
    - The maps compare each group to the norm, not groups to each other: a region found in
      two groups is not evidence that they differ. Groups need to be compared using a separate, direct test.
    - If z-scores were corrected for site with local controls (config.LOCAL_CONTROLS), the local
      controls are at 0 on average by construction, and the other groups are effectively
      compared to them.
    - ROIs in config.ROIS_EXCLUDED_FROM_MAPS (GM of the ventricles, which is noise) are left out.
    - The atlas table names ROIs per hemisphere (lHip_GM_Vol, rHip_GM_Vol), so maps can't be
      drawn for the left/right-averaged model (the table is still printed and saved).

fit_normative_model.py must run first.
"""
import re

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from scipy.stats import binomtest, norm
from statsmodels.stats.multitest import multipletests

import config

MEAN_HEMISPHERES = False
Z_THRESHOLD = 1.96  # extreme = z < -Z_THRESHOLD or z > +Z_THRESHOLD, each direction tested separately
FDR_ALPHA = 0.05
TISSUE_SUFFIX = "_GM_Vol"  # one tissue per map: GM and CSF ROIs (and WM too even if we don't use them in our examples) share the same atlas regions

# Dataset-specific settings -----------------------------------------------------------------
SESSION = None       # value of the "session" column to use; None if the data has a single session
GROUP_COLUMN = None  # column defining the groups; None: all participants as one group
GROUPS = None        # {label: [values of GROUP_COLUMN]} to merge or select groups;
                     # None: one group per value of GROUP_COLUMN
# Examples:
#   RLink (before / after lithium):  SESSION = "M00"; GROUP_COLUMN = "response"
#                                    GROUPS = {"GR": ["GR"], "PaR/NR": ["PaR", "NR"]}
#   OpenNeuro ds004302:              SESSION = None; GROUP_COLUMN = "group"   (HC, AVH-, AVH+)


# ---------------------------------------------------------------------------
# Extreme deviations
# ---------------------------------------------------------------------------
def extreme_deviations(df_z, rois, z_threshold=Z_THRESHOLD, fdr_alpha=FDR_ALPHA):
    """
    Per ROI, % of participants below -z_threshold and above +z_threshold, each tested against
    the rate expected under the norm (one-sided binomial test, FDR-corrected over ROIs and
    both directions together).
    """
    n = len(df_z)
    expected = norm.sf(z_threshold)  # e.g. 2.5% for 1.96 
    # norm.sf is the survival function of the standard normal N(0, 1) (scipy's default):
    # norm.sf(t) = P(z > t) = 1 - CDF(t), the area under the density to the right of t.
    # For t = 1.96 it is 0.025: under the normative distribution, 2.5% of people are above +1.96 by chance
    # alone, and by symmetry 2.5% below -1.96. Each direction is tested against this rate.

    z = df_z[rois].values
    rows = []
    for direction, counts in [("below", (z < -z_threshold).sum(0)), ("above", (z > z_threshold).sum(0))]:
        for roi, k in zip(rois, counts):
            p = binomtest(int(k), n, expected, alternative="greater").pvalue
            rows.append((roi, direction, k, 100 * k / n, p))
    res = pd.DataFrame(rows, columns=["roi", "direction", "n_extreme", "pct", "p"])
    res["p_fdr"] = multipletests(res["p"], method="fdr_bh")[1]
    res["significant"] = res["p_fdr"] < fdr_alpha
    return res.sort_values("pct", ascending=False).reset_index(drop=True)


def select_groups(df_z):
    """{label: rows of df_z} according to GROUP_COLUMN and GROUPS."""
    if GROUP_COLUMN is None:
        return {"All participants": df_z}
    groups = GROUPS or {str(v): [v] for v in sorted(df_z[GROUP_COLUMN].dropna().unique())}
    return {label: df_z[df_z[GROUP_COLUMN].isin(values)] for label, values in groups.items()}


# ---------------------------------------------------------------------------
# Glass brain
# ---------------------------------------------------------------------------
def atlas_label(row, label_column):
    """Value of an ROI (row of the atlas table) in the label image."""
    if label_column == "ROIid":
        return int(row["ROIid"])
    # ROIbaseid holds the atlas label(s); for left ROIs with two ids, the left one is the 2nd
    ids = list(map(int, re.findall(r"\d+", str(row["ROIbaseid"]))))
    return ids[1] if ("Left" in row["ROIname"] and len(ids) == 2) else ids[0]


def plot_glassbrain(roi_values, title="", cmap="Reds", threshold=0, vmax=None):
    """
    Glass-brain plot of per-ROI values on the Neuromorphometrics atlas (config.ATLAS_*).
    roi_values: {ROI name (e.g. "lHip_GM_Vol"): value}. Names are matched to the atlas table's
    ROIabbr with or without their tissue suffix (lHip_GM_Vol or lHip). ROIs not found are skipped.
    """
    atlas = pd.read_csv(config.ATLAS_ROI_TABLE, sep=";")
    ref_im = nib.load(config.ATLAS_LABELS_NII)
    labels = np.asarray(ref_im.dataobj)
    texture = np.zeros(labels.shape)

    for roi, val in roi_values.items():
        base = next((roi[:-len(s)] for s in config.ROI_SUFFIXES if roi.endswith(s)), roi)
        match = atlas[atlas["ROIabbr"] == roi]
        if match.empty:
            match = atlas[atlas["ROIabbr"] == base]
        if match.empty:
            print(f"  {roi} not in the atlas table, not plotted")
            continue
        texture[labels == atlas_label(match.iloc[0], config.ATLAS_LABEL_COLUMN)] = val

    plotting.plot_glass_brain(nib.Nifti1Image(texture, ref_im.affine), display_mode="ortho",
                              colorbar=True, cmap=cmap, plot_abs=True, vmin=0, vmax=vmax, alpha=0.6,
                              threshold=threshold, title=title)
    plotting.show()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_results(res, label, n):
    print(f"\n=== {label} (n={n}): ROIs with more extreme deviations than expected "
          f"({100 * norm.sf(Z_THRESHOLD):.1f}% per direction under the norm) ===")
    sig = res[res["significant"]]
    print(sig.round(4).to_string(index=False) if len(sig) else "none after FDR correction")
    both = sorted(set(sig.loc[sig["direction"] == "below", "roi"]) & set(sig.loc[sig["direction"] == "above", "roi"]))
    if both:  # extreme in both directions: heterogeneous rather than a consistent shift
        print("Significant in both directions (heterogeneous, not a consistent shift):", both)


def plot_maps(res, label, vmax=None):
    """One glass brain below the norm (blue), one above (red), significant ROIs only."""
    sig = res[res["significant"]]
    for direction, cmap, cond in [("below", "Blues", f"z < -{Z_THRESHOLD}"), ("above", "Reds", f"z > {Z_THRESHOLD}")]:
        values = dict(sig.loc[sig["direction"] == direction, ["roi", "pct"]].values)
        if values:
            plot_glassbrain(values, cmap=cmap, vmax=vmax, title=f"{label}: % of participants with {cond}")


if __name__ == "__main__":
    zscores_file = config.zscores_path(MEAN_HEMISPHERES)
    df_z = pd.read_csv(zscores_file)

    # one row per participant: select a session if there are several
    if SESSION is not None:
        df_z = df_z[df_z["session"] == SESSION]
    elif "session" in df_z.columns and df_z["session"].nunique() > 1:
        raise ValueError(f"Several sessions in the data ({sorted(df_z['session'].unique())}): set SESSION")
    assert df_z["participant_id"].is_unique, "participants appear several times: select a session"

    rois = [r for r in config.roi_columns(df_z) if r.endswith(TISSUE_SUFFIX)
            and r not in config.ROIS_EXCLUDED_FROM_MAPS]

    results = {}
    for label, df_group in select_groups(df_z).items():
        results[label] = extreme_deviations(df_group, rois)
        print_results(results[label], label, len(df_group))

    out = pd.concat([r.assign(group=label) for label, r in results.items()], ignore_index=True)
    out_file = zscores_file.replace(".csv", f"_extreme_deviations{TISSUE_SUFFIX}.csv")
    out.to_csv(out_file, index=False)
    print(f"\nSaved results to {out_file}")

    # same color scale for every map, so that groups can be compared
    vmax = max([r.loc[r["significant"], "pct"].max() for r in results.values() if r["significant"].any()], default=None)
    for label, res in results.items():
        plot_maps(res, label, vmax=vmax)
