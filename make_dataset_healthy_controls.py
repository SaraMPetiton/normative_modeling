"""
Step 1: creates the healthy control (HC) dataset (csv file) used to train the normative model.

Pools four cohorts of healthy adults with CAT12 VBM ROI volumes (v12.7, , v12.6 for HCP)
(Neuromorphometrics atlas, 140 GM + 140 CSF = 280 ROIs):
    - OpenBHB  (multi-site, already preprocessed into OpenBHB_roi.csv)
    - UK Biobank
    - HCP      (controls only)
    - OpenMind (OpenNeuro datasets, healthy subjects that passed QC)

For each cohort: keeps healthy subjects with age and sex, one scan per subject
(see filter_openmind_sites), scales ROI volumes so that TIV = 1500 cm3, and harmonizes
sex labels to "female"/"male".

Output: config.HC_DATAFRAME
"""

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np

import config

TARGET_TIV = 1500.0


def scale_by_tiv(df, roi_cols, tiv_col):
    """Rescale ROI volumes so that every subject has TIV = TARGET_TIV."""
    scaling_factor = TARGET_TIV / df[tiv_col]
    df[roi_cols + [tiv_col]] = df[roi_cols + [tiv_col]].mul(scaling_factor, axis=0)
    return df


def roi_columns(df):
    """get list of columns that are roi names in df"""
    return [c for c in df.columns if c.endswith("_GM_Vol") or c.endswith("_CSF_Vol")]


# ---------------------------------------------------------------------------
# Cohorts
# ---------------------------------------------------------------------------
def get_openbhb():
    df = pd.read_csv(config.OPENBHB_ROI)
    rois = roi_columns(df)
    df = df[["participant_id"] + rois + ["age", "sex", "site"]].copy()
    df["cohort"] = "openbhb"
    return df, rois


def get_ukb(rois, verbose=False):
    df = pd.read_csv(config.UKB_ROI, sep="\t").drop(columns=["Unnamed: 0"])
    pheno = pd.read_csv(config.UKB_PHENOTYPE, sep="\t")
    pheno = pheno.rename(
        columns={"31": "sex", # 0 is female and 1 is male
                 "21022": "age" # 21022 = age at recruitment
                 })
    pheno["participant_id"] = "sub-" + pheno["participant_id"].astype(str)

    df = pd.merge(df, pheno[["participant_id", "sex", "age"]], on="participant_id", how="inner")

    # age is age at recruitment -> keep only each subject's first session
    df = (df.sort_values(["participant_id", "session"])
            .drop_duplicates("participant_id", keep="first")
            .reset_index(drop=True))

    assert df["participant_id"].is_unique   # make sure there are no participant_id duplicates

    # UKB codes 0 = female, 1 = male; switch to the convention used here (0 = male, 1 = female).
    # print check:
    if verbose: print(df.groupby("sex")["tiv"].mean()) # checking mean tiv by sex label to confirm men are labelled 1 and women 0

    df["sex"] = df["sex"].replace({0: 1, 1: 0}) # now: 0 = male, 1 = female
    if verbose: print(df.groupby("sex")["tiv"].mean()) # checking it got switched

    df = scale_by_tiv(df, roi_columns(df), "tiv")
    df = df[["participant_id"] + rois + ["age", "sex"]].copy()
    df["site"] = "ukb"
    df["cohort"] = "ukb"
    # plot_age_distribution(df, "ukb")
    return df


def get_hcp(rois):
    df = pd.read_csv(config.HCP_ROI, sep="\t")
    participants = pd.read_csv(config.HCP_PARTICIPANTS, sep="\t")
    participants = participants[participants["diagnosis"] == "control"]

    # HCP sex is already 0 = male, 1 = female
    df = pd.merge(df, participants[["participant_id", "age", "sex", "site"]], on="participant_id", how="inner")
    df = scale_by_tiv(df, roi_columns(df), "TIV")
    df = df[["participant_id"] + rois + ["age", "sex", "site"]].copy()
    df["cohort"] = "hcp"
    return df


def filter_openmind_sites(df):
    """
    Keep one scan per subject (or scans acquired at the same age) for OpenMind datasets
    with several sessions/runs:
        ds005371, ds004656, ds004513, ds004489, ds003416, ds003097:
            keep all (sessions acquired the same day / a few days apart, same age)
        ds005216, ds002995: session 1 only (age corresponds to session 1)
        ds004712: session "NOT1ACH001" (first one chronologically, age at recruitment)
        ds004711, ds004044: remove the single subject with several sessions/runs
        ds004332, ds003639: run 1 only (other runs have head motion)
        ds001417: "before" session
        ds001408: "1preEEG" session
        ds001927: removed entirely
    """
    keep = {
        "ds005216": lambda d: d[d["session"] == "1"], 
        "ds002995": lambda d: d[d["session"] == "1"],
        "ds004712": lambda d: d[d["session"] == "NOT1ACH001"],
        "ds004711": lambda d: d[d["participant_id"].map(d["participant_id"].value_counts()) == 1],
        "ds004044": lambda d: d[d["participant_id"].map(d.groupby("participant_id")["run"].nunique()) == 1],
        "ds004332": lambda d: d[d["run"] == 1],
        "ds003639": lambda d: d[d["run"] == 1],
        "ds001417": lambda d: d[d["session"] == "before"],
        "ds001408": lambda d: d[d["session"] == "1preEEG"],
        "ds001927": lambda d: d.iloc[0:0],
    }
    parts = [keep.get(site, lambda d: d)(d_site) for site, d_site in df.groupby("site", sort=False)]
    df = pd.concat(parts, ignore_index=True)

    # sites where duplicate  participant IDs remain although we did not decide to keep them -> drop the whole site
    sites_keep_all = ["ds005371", "ds003416", "ds003097", "ds004489", "ds004513", "ds004656"]
    dup_sites = df.loc[df["participant_id"].duplicated(keep=False), "site"].unique()
    # sites with duplicate participant IDs :
    dup_sites = [s for s in dup_sites if s not in sites_keep_all]  # ds003639, ds004173, ds004332, ds005264
    return df[~df["site"].isin(dup_sites)].reset_index(drop=True)


def get_openmind(rois):
    participants = pd.read_csv(config.OPENMIND_PARTICIPANTS, sep="\t")
    # participants columns columns ['participant_id', 'age', 'weight', 'bmi', 'sex', 'handedness', 'race', 'health_status']
    # weight, race, and bmi are mostly NaN
    # health_status: [nan 'healthy' 'ill']
    participants = participants[(participants["health_status"] == "healthy") & participants["age"].notna()] # keep healthy controls with age

    # keep participants that passed QC (unique_id = "<dataset>__<subject>__<...>")
    qc = pd.read_csv(config.OPENMIND_QC, sep="\t")  # 113866 subjects
    qc = qc[qc["qc"] == 1] # 112464 subjects
    # create unique_id column with participant_ids extracted from "unique_id" column values
    qc_ids = qc["unique_id"].str.split("__", n=2).str[:2].str.join("__")
    participants = participants[participants["participant_id"].isin(qc_ids)].copy() # keep only participants that passed the QC
    participants["sex"] = participants["sex"].map({"female": 1, "male": 0})

    df = pd.read_csv(config.OPENMIND_ROI, sep="\t") # version cat12 12.7 : 32735 rows/participants;  cat 12.8: 49583 rows 
    df["site"] = df["dataset_id"].astype(str)
    df["participant_id"] = df["dataset_id"].astype(str) + "__sub-" + df["participant_id"].astype(str)
    df = pd.merge(df, participants[["participant_id", "age", "sex"]], on="participant_id", how="inner")

    # deal with duplicates (for participants with multiple sessions or runs, pick one session/run)
    df = filter_openmind_sites(df)

    df = scale_by_tiv(df, roi_columns(df), "tiv")
    df = df[["participant_id"] + rois + ["age", "sex", "site"]].copy()

    # replace ages with more precise ones where available 
    # (not ages from the NeuroSpin saved dataframe but from OpenNeuro's studies' participants.tsv directly)
    precise = []
    for ds, path in config.OPENMIND_PRECISE_AGES.items():
        p = pd.read_csv(path, sep="\t")
        p["participant_id"] = ds + "__" + p["participant_id"].astype(str)
        precise.append(p[["participant_id", "age"]])
    precise = pd.concat(precise).set_index("participant_id")["age"]
    # Update ages: use precise age if available, otherwise keep original
    df["age"] = df["participant_id"].map(precise).fillna(df["age"])

    # remove subjects with missing data (~100 subjects where sex is NaN)
    df = df.dropna().reset_index(drop=True)
    df["cohort"] = "open_mind"
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_hc_dataset():
    """ aggregates all datasets into one dataframe """
    df_openbhb, rois = get_openbhb()
    cohorts = [get_hcp(rois), get_openmind(rois), get_ukb(rois), df_openbhb]
    for c in cohorts:
        print(f"{c['cohort'].iloc[0]}: {len(c)} participants")
        assert c.shape[1] == len(rois) + 5, "wrong number of ROIs (expected CAT12 v12.7, 280 ROIs)"

    df = pd.concat(cohorts, ignore_index=True)
    df["sex"] = df["sex"].map({1.0: "female", 0.0: "male"})  # re-label "sex" to "female" and "male" for easy use of the df
    df = df.dropna().reset_index(drop=True) # remove subjects with missing data
    print(f"Total: {len(df)} participants")
    return df


def describe(df):
    """ 
    prints, for each cohort: the total nb of participants, 
    the mean age (and corresponding standard deviation), and the % of females
    """
    for name, g in list(df.groupby("cohort")) + [("Total", df)]:
        print(f"{name}: N={g['participant_id'].nunique()}, "
              f"age={g['age'].mean():.1f}±{g['age'].std():.1f}, "
              f"female={(g['sex'] == 'female').mean() * 100:.1f}%")


def plot_age_distribution(df, cohort_name="Healthy controls"):
    """
    plots the age distribution of the 'age' column in a DataFrame.
    params
    ----------
    df : dataframe of healthy controls (with 'age' columns)
    cohort_name : name of the cohort for the plot title
    """
    if "age" not in df.columns:
            raise ValueError("The DataFrame must contain an 'age' column.")
    plt.figure(figsize=(8, 5))
    bins = np.linspace(df["age"].min(), df["age"].max(), 41)  # same 40 bins for both sexes
    bin_width = bins[1] - bins[0]
    grid = np.linspace(bins[0], bins[-1], 300)
    for sex, color in [("female", "#E87D8A"), ("male", "#5B8DB8")]:
        ages = df.loc[df["sex"] == sex, "age"]
        plt.hist(ages, bins=bins, alpha=0.7, color=color, edgecolor="black", label=sex.capitalize())
        # density curve, scaled to counts so it sits on the bars
        plt.plot(grid, gaussian_kde(ages)(grid) * len(ages) * bin_width, color=color, linewidth=2.5)
    plt.xlabel("Age")
    plt.ylabel("Number of subjects")
    plt.title(cohort_name + " age distribution", fontsize=20)
    # Make x and y ticks bigger
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Age", fontsize=16)
    plt.ylabel("Number of subjects", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_age_bins_by_cohort(df, bin_width=5):
    """
    for each cohort: bar plot of the number of subjects per age bin (bin_width years).
    useful to visualize with datasets describe which parts of the population age-wise.
    params
    ----------
    df : dataframe of healthy controls (with 'age' and 'cohort' columns)
    bin_width : width of the age bins, in years
    """
    bins = np.arange(df["age"].min() // bin_width * bin_width, df["age"].max() + bin_width, bin_width)
    labels = [f"{int(bins[i])}-{int(bins[i + 1] - 1)}" for i in range(len(bins) - 1)]
    age_bin = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    # number of subjects per age bin and cohort
    count_table = df.groupby([age_bin, df["cohort"]], observed=False).size().unstack(fill_value=0)

    ax = count_table.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_ylim(0, count_table.values.max() * 1.15)
    for p in ax.patches:  # count above each bar
        if p.get_height() > 0:
            ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 1, str(int(p.get_height())),
                    ha="center", va="bottom", fontsize=10, rotation=90)

    plt.xlabel("Age bin (years)", fontsize=16)
    plt.ylabel("Number of subjects", fontsize=16)
    plt.title(f"Number of subjects per {bin_width}-year age bin by cohort", fontsize=20)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Cohort", fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_hc = generate_hc_dataset()
    df_hc.to_csv(config.HC_DATAFRAME, index=False)
    describe(df_hc)
    plot_age_distribution(df_hc)
