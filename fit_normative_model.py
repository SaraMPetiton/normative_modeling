"""
Step 2: fits the normative model on healthy controls and computes z-scores.

    1. Loads the HC dataset (make_dataset_healthy_controls.py) and splits it 90/10 (train/test),
       stratified on age decile, sex and site. We choose a 90/10 split because of the large
       dataset size (N>50k), meaning enough participants are part of the testing set.
    2. Fits the normative model (by default NormativeBLR: warped, B-spline on age; set
       MODEL_CLASS / MODEL_PARAMS in config.py to use another one) on the train set, checks
       on the test set that R² is reasonable and z-scores are ~N(0, 1). Saves the model to
       models/. If the model already exists in models/, it is loaded instead of refitted.
    3. Computes z-scores of new participants (config.NEW_PARTICIPANTS) using the normative
       model, and saves them to results/zscores/. Participants can have two or more time
       points, in which case the csv has one row per participant and session, with an
       additional column "session".
    4. Optionally, z-scores can be adjusted for site effects if enough healthy controls
       from the same site are available for calibration (config.LOCAL_CONTROLS): at each
       site, the mean z-score of its controls serves as an offset, subtracted from the
       z-scores of all the site's participants (see adjust_for_site).
       ! The site's controls are then at 0 on average by construction: don't use their
       z-scores in analyses that need them to be independent of the other groups'. !
       We chose this over hierarchical Bayesian regression (HBR; Kia et al., 2020, MICCAI;
       Kia et al., 2022, PLOS ONE) because HBR needs the site of every training scan, which
       very large pooled datasets often don't have, and needs the model to be refitted on
       each new site's data. 
       "Site" is also a vague proxy for what actually causes the effects: a combination
       of scanner model, field strength, resolution, sequence, software version, etc., so
       modelling "site" in training would capture site effects only partially, and could
       introduce bias where site is confounded with other variables, such as age when an
       age range is covered by a single cohort (e.g. UK Biobank above ~70 in our training
       data).

ROIs: Neuromorphometrics atlas, CAT12 v12.7 (280 ROIs: 140 GM + 140 CSF), see config.py.

Set MEAN_HEMISPHERES = True to use the version of the model where left and right ROIs are
averaged: 138 ROIs instead of 140, because the 4 ROIs that have no left/right counterpart
(276 of the 280 ROIs form 138 pairs) are dropped.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import config

MEAN_HEMISPHERES = False
# The mean z-scores of config.ROIS_OF_INTEREST are printed after computing them,
# per session and per group if GROUP_COLUMN is set.
GROUP_COLUMN = "response"  # column of the new participants to group by when printing (None: no grouping)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def mean_left_right_hemispheres(df, verbose=False):
    """Replaces each pair of ROIs lX / rX by their mean, named X. ROIs without a pair are dropped."""
    rois = config.roi_columns(df)
    other_cols = [c for c in df.columns if c not in rois]
    averaged, paired = {}, set()
    for lcol in [c for c in rois if c.startswith(config.LEFT_PREFIX)]:
        base = lcol[len(config.LEFT_PREFIX):]
        rcol = config.RIGHT_PREFIX + base
        if rcol in rois:
            averaged[base] = df[[lcol, rcol]].mean(axis=1)
            paired.update([lcol, rcol])
    if verbose:
        print("ROIs without a left/right pair (dropped):", [r for r in rois if r not in paired])
    return pd.concat([df[other_cols], pd.DataFrame(averaged, index=df.index)], axis=1)


def load_hc(mean_hemispheres=False):
    df = pd.read_csv(config.HC_DATAFRAME)
    if mean_hemispheres:
        df = mean_left_right_hemispheres(df, verbose=True)
    rois = config.roi_columns(df)
    n_expected = config.N_ROIS_MEAN_HEMISPHERES if mean_hemispheres else config.N_ROIS
    assert len(rois) == n_expected, f"{len(rois)} ROIs found, {n_expected} expected (see config.py)"
    return df, rois


def load_new_participants(mean_hemispheres=False):
    df = pd.read_csv(config.NEW_PARTICIPANTS)
    if mean_hemispheres:
        df = mean_left_right_hemispheres(df)
    return df


def split_train_test(df, test_size=0.1, seed=42, q_age=10, min_site_size=600, target_bin_size=9000, verbose=False):
    """
    Train/test split stratified on age x sex x site (sites with fewer than
    min_site_size subjects are pooled as "other"). This is the split used for the saved models.
    q_age: int = 10 # how many bins the age column is divided into for stratification

    Each age bin is capped at target_bin_size subjects. Since qcut makes bins of equal size
    (~5,400 subjects each for the ~54,000 HC), the cap removes no one: every HC is used.
    The per-bin sampling step is kept anyway because it shuffles the rows, and the split
    depends on that order: removing it would give a different split.
    """

    df = df.copy()
    df["age_bin"] = pd.qcut(df["age"], q=q_age, duplicates="drop")
    site_counts = df["site"].value_counts()
    site_binned = df["site"].replace(site_counts[site_counts < min_site_size].index, "other")
    df["strata"] = df["age_bin"].astype(str) + "_" + df["sex"].astype(str) + "_" + site_binned.astype(str)

    df = df.groupby("age_bin", group_keys=False, observed=False)[list(df.columns)].apply(
        lambda x: x.sample(min(target_bin_size, len(x)), random_state=seed))

    df_train, df_test = train_test_split(df, test_size=test_size, stratify=df["strata"], random_state=seed)

    if verbose:
        print("Balanced dataset:")
        print("Train size:", len(df_train), "Test size:", len(df_test))
        print("Samples per age bin:", df['age_bin'].value_counts().to_dict())

    return df_train.drop(columns=["age_bin", "strata"]), df_test.drop(columns=["age_bin", "strata"])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def fit_normative_model(df_hc, rois, model_class=None, model_params=None, evaluate=True):
    """Fits a normative model (default: config.MODEL_CLASS(**config.MODEL_PARAMS)) on the HC train set."""
    model_class = model_class or config.MODEL_CLASS
    model_params = config.MODEL_PARAMS if model_params is None else model_params

    df_train, df_test = split_train_test(df_hc)
    X_tr, y_tr = df_train[["age", "sex"]], df_train[rois].values
    X_te, y_te = df_test[["age", "sex"]], df_test[rois].values
    print(f"Fitting {model_class.__name__} on {len(df_train)} HC, testing on {len(df_test)} HC, {len(rois)} ROIs")

    model = model_class(**model_params)
    model.fit(X_tr, y_tr)

    if evaluate:
        pred_te, z_te = model.predict(X_te, y_te, return_zscores=True)
        r2_per_roi = [r2_score(y_te[:, i], pred_te[:, i]) for i in range(len(rois))]
        print(f"R² train: {r2_score(y_tr, model.predict(X_tr)):.3f} | test: {r2_score(y_te, pred_te):.3f}")
        print(f"Per-ROI test R²: mean {np.mean(r2_per_roi):.3f}, "
              f"min {np.min(r2_per_roi):.3f}, max {np.max(r2_per_roi):.3f}")
        # on held-out HC, z-scores should be ~N(0, 1)
        print(f"Test z-scores: mean {z_te.mean():.3f} (should be ~0), std {z_te.std():.3f} (should be ~1)")
        print(f"Per-ROI z mean range [{z_te.mean(0).min():.3f}, {z_te.mean(0).max():.3f}], "
              f"std range [{z_te.std(0).min():.3f}, {z_te.std(0).max():.3f}]")
    return model


def load_or_fit_model(mean_hemispheres=False):
    path = config.model_path(mean_hemispheres)
    df_hc, rois = load_hc(mean_hemispheres)
    if os.path.exists(path):
        print(f"Loading {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
    else:
        model = fit_normative_model(df_hc, rois)
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
    return model, rois


# ---------------------------------------------------------------------------
# Z-scores
# ---------------------------------------------------------------------------
def compute_zscores(model, rois, df):
    """
    Z-scores of every ROI for the rows of df (participant_id, age, sex, ROIs, other columns).
    Returns a dataframe with participant_id, the z-scores (same ROI order as the model),
    then all non-ROI columns of df (age, sex, session, site, ...).
    """
    missing = [r for r in rois if r not in df.columns]
    assert not missing, f"{len(missing)} ROIs of the model are missing from the new participants, e.g., {missing[:5]}"
    X = df[["age", "sex"]].astype({"age": float})  # sex must be "female"/"male"
    _, z = model.predict(X, df[rois].values, return_zscores=True)
    df_z = pd.DataFrame(z, columns=rois, index=df.index)
    other_cols = [c for c in df.columns if c not in config.roi_columns(df) and c != "participant_id"]
    return pd.concat([df[["participant_id"]], df_z, df[other_cols]], axis=1)

def adjust_for_site(df_z, rois, is_control, site_col="site", min_controls=30):
    """
    Corrects z-scores for potential site effects using healthy controls scanned at the same site (when available).

    The normative model was fitted on very large amounts of data from many scanners, so
    that its norm doesn't depend on any one of them. 
    
    The new data one might want to evaluate against the normative model
    may have more or less site artefacts, which has the potential to shift all of its z-scores.

    At each site, the local controls' mean z-score per ROI
    estimates that shift (controls should be at 0 on average), and it is subtracted from
    every participant of the site:  z_adjusted = z - mean(z of the site's controls).
    This assumes the site offset is the same at every age and for both sexes (sex is already
    in the model, so the controls don't need to be sex-balanced). Local controls are then at
    0 on average, so the other participants are effectively compared to them.

    Sites with fewer than min_controls controls are left unchanged: the offset is estimated
    with a standard error of about SD / sqrt(n), and every participant of the site inherits
    that error (~0.2 SD with 30 controls).
    A column "site_adjusted" (True/False) records which rows were corrected.

    df_z : z-scores (one row per participant and session), with a site column
    rois : ROI columns to correct
    is_control : boolean array, True for the rows of local healthy controls
    """

    df_z = df_z.copy()
    is_control = np.asarray(is_control, dtype=bool)
    df_z["site_adjusted"] = False
    for site, n in df_z.loc[is_control, site_col].value_counts().items():
        if n < min_controls:
            print(f"Site {site}: {n} local controls (< {min_controls}), not corrected")
            continue
        offset = df_z.loc[is_control & (df_z[site_col] == site).values, rois].mean()
        rows = (df_z[site_col] == site).values
        df_z.loc[rows, rois] = df_z.loc[rows, rois] - offset
        df_z.loc[rows, "site_adjusted"] = True
        print(f"Site {site}: corrected with {n} local controls "
              f"(mean |offset| over ROIs: {offset.abs().mean():.2f} SD)")
    no_controls = set(df_z[site_col]) - set(df_z.loc[is_control, site_col])
    if no_controls:
        print(f"Sites without local controls, not corrected: {sorted(no_controls)}")
    return df_z

def save_zscores(mean_hemispheres=False):
    model, rois = load_or_fit_model(mean_hemispheres)
    df = load_new_participants(mean_hemispheres)
    df_z = compute_zscores(model, rois, df)

    if config.LOCAL_CONTROLS is not None:  # site correction with local healthy controls
        column, value = config.LOCAL_CONTROLS
        df_z = adjust_for_site(df_z, rois, (df_z[column] == value).values,
                               site_col=config.SITE_COLUMN, min_controls=config.MIN_CONTROLS_PER_SITE)

    os.makedirs(os.path.dirname(config.zscores_path(mean_hemispheres)), exist_ok=True)
    df_z.to_csv(config.zscores_path(mean_hemispheres), index=False)
    print(f"Saved z-scores of {df['participant_id'].nunique()} participants "
          f"({len(df)} rows) to {config.zscores_path(mean_hemispheres)}")

    # quick look at a few ROIs
    rois_to_print = [r for r in config.rois_of_interest(mean_hemispheres) if r in rois]
    if rois_to_print:
        # if there is a "session" column and/or "group" column, groups mean z-scores by session and group:
        # for example, for the RLiNK project, we had participants' data before and after treatment (session M00 and M03),
        # and for two groups: good responders to treatment, and non- or partial- responders
        by = [c for c in ["session", GROUP_COLUMN] if c and c in df_z.columns]
        print("\nMean z-scores" + (f" by {' and '.join(by)}:" if by else ":"))
        print((df_z.groupby(by)[rois_to_print].mean().T if by else df_z[rois_to_print].mean()).round(3))
    return df_z, model


if __name__ == "__main__":
    save_zscores(mean_hemispheres=MEAN_HEMISPHERES)
