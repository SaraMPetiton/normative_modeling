"""
All settings shared by the scripts: atlas / ROI conventions, normative model, and paths.
Edit this file to use your own data.

Atlas: Neuromorphometrics, as computed by CAT12 v12.7 (VBM ROI volumes, cm3):
140 gray matter (GM) + 140 CSF ROIs = 280 ROIs. To use another CAT12 version, update the
"Atlas / ROIs" section below (number of ROIs, how ROI columns are named); the rest of the
code does not change, but the normative models in models/ must be refitted, since they
predict exactly the 280 CAT12 v12.7 ROIs, in a fixed order.
"""
import os

from normative_models import NormativeBLR

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_DIR, "data")            # input dataframes (not versioned)
MODEL_DIR = os.path.join(REPO_DIR, "models")         # fitted normative models (.pkl)
RESULTS_DIR = os.path.join(REPO_DIR, "results")      # z-scores, figures (not versioned)

# ---------------------------------------------------------------------------
# Atlas / ROIs (CAT12 v12.7, Neuromorphometrics)
# ---------------------------------------------------------------------------
CAT12_VERSION = "12.7"
ROI_SUFFIXES = ("_GM_Vol", "_CSF_Vol")   # a column is an ROI if its name ends with one of these
LEFT_PREFIX, RIGHT_PREFIX = "l", "r"     # hemisphere prefix of ROI names (e.g. lHip_GM_Vol / rHip_GM_Vol)
N_ROIS = 280                             # 140 GM + 140 CSF
# With left/right averaged: each pair lX / rX becomes one ROI X. 276 of the 280 ROIs form
# 138 pairs; the 4 ROIs without a left/right counterpart are dropped, hence 138 ROIs and
# not 140 (see fit_normative_model.mean_left_right_hemispheres, which prints them).
N_ROIS_MEAN_HEMISPHERES = 138


# ROIs of interest, printed and plotted by the scripts. Names follow the CAT12 v12.7 format:
# this will need an update if a new version names ROIs differently. Their left/right-averaged names
# are derived from these (see rois_of_interest), so there is only this list to change.
ROIS_OF_INTEREST = ["lHip_GM_Vol", "rHip_GM_Vol",   # hippocampus
                    "lAmy_GM_Vol", "rAmy_GM_Vol",   # amygdala
                    "lPal_GM_Vol", "rPal_GM_Vol",   # pallidum
                    "lPut_GM_Vol", "rPut_GM_Vol"]   # putamen


def roi_columns(df):
    """ROI columns of a dataframe, in their order in the dataframe."""
    return [c for c in df.columns if c.endswith(ROI_SUFFIXES)]


def mean_hemisphere_name(roi):
    """Name of an ROI once left and right are averaged: lHip_GM_Vol -> Hip_GM_Vol."""
    for prefix in (LEFT_PREFIX, RIGHT_PREFIX):
        if roi.startswith(prefix):
            return roi[len(prefix):]
    return roi


def rois_of_interest(mean_hemispheres=False):
    """ROIS_OF_INTEREST, or their left/right-averaged names (without duplicates, same order)."""
    if not mean_hemispheres:
        return list(ROIS_OF_INTEREST)
    return list(dict.fromkeys(mean_hemisphere_name(r) for r in ROIS_OF_INTEREST))


# ---------------------------------------------------------------------------
# Normative model
# ---------------------------------------------------------------------------
# Any class can be used if it implements:
#   fit(X, Y)                                  X: DataFrame [age, sex], Y: array (n_subjects, n_rois)
#   predict(X, Y=None, return_zscores=False)   -> means (n_subjects, n_rois) [, z-scores]
#   stds_                                      per-ROI SD (used to plot normative curves)
MODEL_CLASS = NormativeBLR
MODEL_PARAMS = dict(warp=True, bsplines=True, n_knots=3)
MODEL_NAME = "normative_BLR_BigHC_janv2026_280rois"   # file name of the saved model, without .pkl


def model_name(mean_hemispheres=False):
    """Name of the saved normative model (and prefix of its z-score files)."""
    return MODEL_NAME + ("_mean_btw_hemispheres" if mean_hemispheres else "")


def model_path(mean_hemispheres=False):
    return os.path.join(MODEL_DIR, model_name(mean_hemispheres) + ".pkl")


# ---------------------------------------------------------------------------
# Healthy control (HC) dataset, built by make_dataset_healthy_controls.py
# ---------------------------------------------------------------------------
HC_DATAFRAME = os.path.join(DATA_DIR, "hcp_open_mind_ukb_openbhb_roi_vbm_with_age_with_participant_id_280rois.csv")

# Raw cohort files used to build the HC dataset (original NeuroSpin locations)
OPENBHB_ROI = os.path.join(DATA_DIR, "OpenBHB_roi.csv")
NS_PSY="/neurospin/psy/"

UKB_ROI = NS_PSY+"ukb/derivatives/soft-cat12_provider-ns_ver-7.r1743_vbmroi/cat12_vbm_roi.tsv"
UKB_PHENOTYPE = NS_PSY+"ukb/phenotypes/ses-0/baseline-characteristics-population-characteristics.tsv"

HCP_ROI = NS_PSY+"hcp/derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv"
HCP_PARTICIPANTS = NS_PSY+"hcp/HCP_t1mri_mwp1_participants_v-2023.csv"

OPENMIND_ROI = NS_PSY+"openmind/derivatives/cat12-12.7_vbm_roi/neuromorphometrics_cat12_vbm_roi.tsv"
OPENMIND_PARTICIPANTS = NS_PSY+"openmind/participants.tsv"
OPENMIND_QC = NS_PSY+"openmind/partial_rawdata_qc.tsv"
# more precise ages than those in OPENMIND_PARTICIPANTS for two OpenNeuro datasets
OPENMIND_PRECISE_AGES = {
    "ds003097": os.path.join(DATA_DIR, "participants_ds003097.tsv"),
    "ds004711": os.path.join(DATA_DIR, "participants_ds004711.tsv"),
}

# ---------------------------------------------------------------------------
# New participants to evaluate against the norm
# ---------------------------------------------------------------------------
# csv with one row per participant (and per session if longitudinal), with columns
# participant_id, age, sex ("female"/"male"), the ROIs (TIV-scaled to 1500, same naming as
# the HC data), optionally "session", plus any other columns (kept in the z-score files).
# Example: RLink, bipolar patients before (M00) and 3 months after (M03) lithium.
NEW_PARTICIPANTS = os.path.join(DATA_DIR, "df_ROI_age_sex_site_M00_M03_v4labels_280rois_cat12_7.csv")
NEW_PARTICIPANTS_NAME = "rlink"          # used to name the z-score file


def zscores_path(mean_hemispheres=False):
    """z-score csv of the new participants (one row per participant and session)."""
    return os.path.join(RESULTS_DIR, "zscores",
                        f"zscores_{NEW_PARTICIPANTS_NAME}_{model_name(mean_hemispheres)}.csv")


# ---------------------------------------------------------------------------
# Atlas images (for glass-brain plots)
# ---------------------------------------------------------------------------
ATLAS_LABELS_NII = "/drf/local/spm12/tpm/labels_Neuromorphometrics.nii"
ATLAS_ROI_TABLE = os.path.join(DATA_DIR, "lobes_Neuromorphometrics_with_dfROI_correspondencies.csv")
