import numpy as np
import nibabel as nib
from nilearn import plotting
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re, os
import pandas as pd

ROOT="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat_backup/"
ZSCORES_PATH=ROOT+"reports/normative_modeling/"
ATLAS_DF = pd.read_csv(ROOT+"data/processed/lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=';')
VOL_FILE_VBM = "/drf/local/spm12/tpm/labels_Neuromorphometrics.nii"

def plot_glassbrain_VBM_ROI(dict_plot=None, title="", threshold=20): 
    """
        Aim : plot glassbrain of specfic ROI from covariance values obtained with an SVM-RBF and VBM ROI features
    """    
    matching_rows = ATLAS_DF[ATLAS_DF["ROIabbr"].isin(list(dict_plot.keys()))]
    roi_names = matching_rows["ROIname"].tolist()

    ref_im = nib.load(VOL_FILE_VBM)
    ref_arr = ref_im.get_fdata()
    # labels = sorted(set(np.unique(ref_arr).astype(int))- {0}) # 136 labels --> 'Left Inf Lat Vent', 
    # 'Right vessel', 'Left vessel' missing in data
    texture_arr = np.zeros(ref_arr.shape, dtype=float)
    
    for name, val in dict_plot.items():
        # each baseid is the number associated to the ROI in the nifti image
        baseids = ATLAS_DF[(ATLAS_DF['ROIabbr'] == name)]["ROIbaseid"].values
        name = ATLAS_DF[(ATLAS_DF['ROIabbr'] == name)]["ROIname"].values
        name = name[0]

        int_list = list(map(int, re.findall(r'\d+', baseids[0])))
        if "Left" in name: 
            if len(int_list)==2: baseid = int_list[1]
            else : baseid = int_list[0]
        else : 
            baseid = int_list[0]
        texture_arr[ref_arr == baseid] = val

    print(np.shape(texture_arr))

    cmap = "viridis" #plt.cm.coolwarm #"viridis"
    vmin = 0
    vmax = np.max(texture_arr)
    print("vmin vmax texture arr", np.min(texture_arr),"     ",vmax)
    texture_im = nib.Nifti1Image(texture_arr, ref_im.affine)

    plotting.plot_glass_brain(
        texture_im,
        display_mode="ortho",
        colorbar=True,
        cmap=cmap,
        plot_abs=False ,
        alpha = 0.6,
        threshold=threshold, 
        title=title)
    plotting.show() 



def get_empirical_score_threshold_GR_vs_NR(df0_gr, df3_gr, df0_nr_par, df3_nr_par, 
                                            n_permutations=10000, alpha=0.01, normalization=False, change=False):
    """
    either for the normalization or change score (normalization (bool) and change(bool) parameters),
    find a threshold for those scores, separating GR from PaR/NR correctly through permutaiton testing of class labels
    """
    assert normalization or change

    # pooling GR and NR/PaR subjects together
    df0_all = pd.concat([df0_gr, df0_nr_par], ignore_index=True)
    df3_all = pd.concat([df3_gr, df3_nr_par], ignore_index=True)
    
    n_gr = len(df0_gr[df0_gr["participant_id"].isin(df3_gr["participant_id"])])
    
    list_rois = [r for r in df0_all.columns.to_list() if r.endswith("_GM_Vol")
                 and "WM" not in r and "CSF" not in r]

    df0_common = df0_all[df0_all["participant_id"].isin(df3_all["participant_id"])]
    df0_common = df0_common.set_index("participant_id").loc[df3_all["participant_id"].values.tolist()].reset_index()
    df3_common = df3_all.reset_index(drop=True)

    assert all(df0_common["participant_id"] == df3_common["participant_id"])

    def compute_score(z0, z3):

        if normalization: 
            normalization_pct = (np.abs(z3) <= np.abs(z0)).mean() * 100
            mask = np.abs(z3) <= np.abs(z0)
            delta_abs = (np.abs(z0) - np.abs(z3))[mask].mean() if mask.sum() > 0 else 0
            score = np.abs(normalization_pct * delta_abs)

        if change: 
            delta_abs = (np.abs(z0) - np.abs(z3)).mean()

            if delta_abs > 0: # subjects are changed after medication
                change_pct = (np.abs(z3) < np.abs(z0)).mean()* 100
            else: change_pct = (np.abs(z3) > np.abs(z0)).mean()* 100       
            score = np.abs(change_pct * delta_abs)

        return score

    # observed difference in normalization score between GR and NR/PaR
    obs_diff = {}
    for roi in list_rois:
        z0 = df0_common[roi].values
        z3 = df3_common[roi].values
        score_gr = compute_score(z0[:n_gr], z3[:n_gr])
        score_nr = compute_score(z0[n_gr:], z3[n_gr:])
        obs_diff[roi] = score_gr - score_nr

    # build global null distribution by permuting group labels
    global_null = []
    n_total = len(df0_common)

    for _ in range(n_permutations):
        # randomly assign n_gr subjects to "GR", rest to "NR" or "PaR"
        perm_idx = np.random.permutation(n_total)
        idx_gr = perm_idx[:n_gr]
        idx_nr_par = perm_idx[n_gr:]

        for roi in list_rois:
            z0 = df0_common[roi].values
            z3 = df3_common[roi].values
            score_gr_perm = compute_score(z0[idx_gr], z3[idx_gr])
            score_nr_perm = compute_score(z0[idx_nr_par], z3[idx_nr_par])
            global_null.append(np.abs(score_gr_perm - score_nr_perm))

    # threshold at (1 - alpha) percentile of global null
    tau_S = np.percentile(global_null, (1 - alpha) * 100)
    significant_rois = {roi: diff for roi, diff in obs_diff.items()
                        if np.abs(diff) > tau_S}
    print(f"Empirical threshold for GR vs NR/PaR score difference (tau_S) at alpha={alpha}: {tau_S:.2f}")

    return significant_rois, tau_S


def get_dict_normalization_from_zscores(df0, df3):
    """
    get a score of "normalization" for each ROI across RLink subjects
    quantifies how much the deviation from the norm decreases after Li intake
    formula = for ROIs with over 65% of subjects showing a closening towards the healthy norm, 
    (%)*mean(|z3|-|z0|)*sign(increase or decrease of GM volume)
    """

    # Computing % of subjects per ROI with |z| > 2
    list_rois = [r for r in df0.columns.to_list() if r.endswith("_GM_Vol")]

    # Keep only participants with M0 and M3 measures
    df0_common = df0[df0["participant_id"].isin(df3["participant_id"])]
    assert df0_common["participant_id"].is_unique

    # Align on participant_id to ensure matching order
    df0_common = df0_common.set_index("participant_id").loc[df3["participant_id"].values.tolist()].reset_index()
    df3_common = df3.reset_index(drop=True)
    # print(df0_common)
    # print(df3_common)

    assert all(df0_common["participant_id"] == df3_common["participant_id"])
    assert all(df0_common["response"] == df3_common["response"])

    dict_plot = {}
    rows = []

    for roi in list_rois:
        if "WM" not in roi and "CSF" not in roi:

            z0 = df0_common[roi]
            z3 = df3_common[roi]

            # "normalization"/closening toward norm --> equivalent to a closening to the "norm" after Li intake
            # percentage of subjects that get closer to the norm after medication
            normalization_pct = (z3.abs() <= z0.abs()).mean()* 100

            # for those that get closer to the norm, how much closer do they get?
            # magnitude of improvement
            # mean magnitude across subjects that normalize
            delta_abs = (z0.abs() - z3.abs())[z3.abs() <= z0.abs()].mean()

            # direction of change --> increase or decrease of GM volume after medication
            changeGM = z3.mean() - z0.mean()
            direction = np.sign(changeGM)

            # improvement score
            # improvement_score = np.abs(normalization_pct * delta_abs) * direction * (normalization_pct > 50)
            improvement_score = np.abs(normalization_pct * delta_abs) * direction * (normalization_pct > 50)

            if abs(improvement_score) > 23.3: #37.67: #23.3:
                rows.append((roi, normalization_pct, delta_abs, direction, improvement_score, changeGM))
            dict_plot[roi] = improvement_score

    # sorting by descending absolute improvement score
    rows.sort(key=lambda x: abs(x[4]), reverse=True)

    for roi, normalization_pct, delta_abs, direction, improvement_score, changeGM in rows:
        print(
            f"{roi:30s} , "
            f"% of subjects that normalize  = {normalization_pct:5.1f} , "
            f"normalization magnitude = {delta_abs:6.3f} , "
            f"direction of GM change = {direction:+.0f} , "
            f"normalization score = {improvement_score:7.2f} , "
            f"magnitude of GM change = {changeGM:6.3f}"
        )

    return dict_plot


def get_dict_change_from_zscores(df0, df3):
    """
    get a score of "change" for each ROI across RLink subjects
    quantifies how much the distance from a healthy norm changes from before to after Li intake
    --> different from normalization as some regions may show such an increase in GM Vol that the 
    neuroanatomy shifts farther from the healthy norm instead of closer

    formula = for ROIs with over 65% of subjects showing a closening towards the healthy norm, 
    (%)*mean(|z3|-|z0|)*sign(increase or decrease of GM volume)
    """
    # Computing % of subjects per ROI with |z| > 2
    list_rois = [r for r in df0.columns.to_list() if r.endswith("_GM_Vol")]

    # Keep only participants with M0 and M3 measures
    df0_common = df0[df0["participant_id"].isin(df3["participant_id"])]
    assert df0_common["participant_id"].is_unique

    # Align on participant_id to ensure matching order
    df0_common = df0_common.set_index("participant_id").loc[df3["participant_id"].values.tolist()].reset_index()
    df3_common = df3.reset_index(drop=True)
    # print(df0_common)
    # print(df3_common)

    assert all(df0_common["participant_id"] == df3_common["participant_id"])
    assert all(df0_common["response"] == df3_common["response"])

    dict_plot = {}
    rows = []

    for roi in list_rois:
        if "WM" not in roi and "CSF" not in roi:

            z0 = df0_common[roi]
            z3 = df3_common[roi]

            # how much change is there between m0 and m3, with regards to a healthy norm, after accounting for age and sex?
            # mean magnitude across subjects
            delta_abs = (z0.abs() - z3.abs()).mean()

            # if, for the current ROI, subjects are closer to the norm after lithium intake, 
            # what percentage of subjects are they?
            # if the mean across subjects, delta_abs>=0, what is the % of subjects for which it is the case?

            if delta_abs > 0: # subjects are changed after medication
                change_pct = (z3.abs() < z0.abs()).mean()* 100
            else: change_pct = (z3.abs() > z0.abs()).mean()* 100

            # direction of change --> increase or decrease of GM volume after medication
            changeGM = z3.mean() - z0.mean()
            direction = np.sign(changeGM)

            # change score
            change_score = np.abs(change_pct * delta_abs) * direction

            if abs(change_score) > 0: #24.77:
                rows.append((roi, change_pct, delta_abs, direction, change_score, changeGM))

            dict_plot[roi] = change_score

    # sorting by descending absolute change score
    rows.sort(key=lambda x: abs(x[4]), reverse=True)

    for roi, change_pct, delta_abs, direction, change_score, changeGM in rows:
        print(
            f"{roi:30s} , "
            f"% of subjects that change  = {change_pct:5.1f} , "
            f"change magnitude = {delta_abs:6.3f} , "
            f"direction of GM change = {direction:+.0f} , "
            f"change score = {change_score:7.2f} , "
            f"magnitude of GM change = {changeGM:6.3f}"
        )

    return dict_plot

def main():
    # dict_tval = {'lHip_GM_Vol':2.81,'rHip_GM_Vol':2.81,'lAmy_GM_Vol':3.45, 'rAmy_GM_Vol':3.45}

    # plot_glassbrain_VBM_ROI(dict_tval, " ",threshold=0)
    # quit()

    modelname="normative_BLR_BigHC_janv2026_280rois"
    pathzscoresM0 = ZSCORES_PATH+"df_zscores_M0_with_age_sex_site_"+modelname+"_cat12_7_280ROIs.csv"
    pathzscoresM3 = ZSCORES_PATH+"df_zscores_M3_with_age_sex_site_"+modelname+"_cat12_7_280ROIs.csv"
    if os.path.exists(pathzscoresM0): df_zscores_M0_with_age_sex_site = pd.read_csv(pathzscoresM0)
    if os.path.exists(pathzscoresM3): df_zscores_M3_with_age_sex_site = pd.read_csv(pathzscoresM3)

    df_zscores_M0_with_age_sex_site = df_zscores_M0_with_age_sex_site[df_zscores_M0_with_age_sex_site["sex"]=="male"].reset_index()
    df_zscores_M3_with_age_sex_site = df_zscores_M3_with_age_sex_site[df_zscores_M3_with_age_sex_site["sex"]=="male"].reset_index()

    dfM0_zscores_GR = df_zscores_M0_with_age_sex_site[df_zscores_M0_with_age_sex_site["response"]==1]
    dfM0_zscores_NRPaR = df_zscores_M0_with_age_sex_site[df_zscores_M0_with_age_sex_site["response"]==0]

    dfM3_zscores_GR = df_zscores_M3_with_age_sex_site[df_zscores_M3_with_age_sex_site["response"]==1]
    dfM3_zscores_NRPaR = df_zscores_M3_with_age_sex_site[df_zscores_M3_with_age_sex_site["response"]==0]

    # find threshold for normative scores that selects significant ROIs based on differentiating GR from NR/PaR
    # get_empirical_score_threshold_GR_vs_NR(dfM0_zscores_GR, dfM3_zscores_GR, dfM0_zscores_NRPaR, dfM3_zscores_NRPaR, change=True)
    # Empirical threshold for GR vs NR/PaR score difference (tau_S) at alpha=0.01: 23.3 for NORMATIVE SCORE
    # Empirical threshold for GR vs NR/PaR score difference (tau_S) at alpha=0.01: 20.82 for CHANGE SCORE

    # print("GR")
    # resGR=get_dict_normalization_from_zscores(dfM0_zscores_GR, dfM3_zscores_GR)
    # print("\nNR PaR")
    # resNRPaR=get_dict_normalization_from_zscores(dfM0_zscores_NRPaR, dfM3_zscores_NRPaR)
    # plot_glassbrain_VBM_ROI(resGR,"zscores GR",threshold=23.3)
    # plot_glassbrain_VBM_ROI(resNRPaR, "zscores NR/PaR",threshold=23.3)
    

    print("GR")
    resGR_change = get_dict_change_from_zscores(dfM0_zscores_GR, dfM3_zscores_GR)
    print("\nNR PaR")
    resNRPaR_change=get_dict_change_from_zscores(dfM0_zscores_NRPaR, dfM3_zscores_NRPaR)
    plot_glassbrain_VBM_ROI(resGR_change,"zscores GR",threshold=20.82)
    plot_glassbrain_VBM_ROI(resNRPaR_change, "zscores NR/PaR",threshold=20.82)


if __name__ == "__main__":
    main()