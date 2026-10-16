import numpy as np
import pandas as pd
import sys, json, pickle, os
from normative_models import NormativeBLR

# to fit the normative model on OpenBHB data
sys.path.append("/neurospin/signatures/2025_spetiton_rlink_predict_response_anat_backup/")
from utils import stratified_split, get_lists_roi_in_both_openBHB_and_rlink, stratified_split_balanced_age

# to evaluate normative model
from sklearn.metrics import r2_score

# zscores classification
import sklearn.linear_model as lm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#plots
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# residualization
sys.path.append('/neurospin/signatures/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer
# from nitk.ml_utils.residualization import get_residualizer

# inputs
ROOT="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat_backup/"
DATA_DIR=ROOT+"data/processed/"
OPENBHB_DATAFRAME = DATA_DIR+"OpenBHB_roi.csv"
# DF_PATH_HEALTHY_BRAINS = DATA_DIR + "hcp_open_mind_ukb_openbhb_roi_vbm_with_age_with_participant_id.csv"
DF_PATH_HEALTHY_BRAINS = DATA_DIR + "hcp_open_mind_ukb_openbhb_roi_vbm_with_age_with_participant_id_280rois.csv"
DF_PATH_HEALTHY_BRAINS_NO_BIOBD_BSNIP_HC = DATA_DIR + "hcp_open_mind_ukb_openbhb_roi_vbm_with_age_with_participant_id_280rois_without_biobdbsnip_HC.csv"

# RLink inputs
RLINK_DATAFRAME_M00_M03 = DATA_DIR + "df_ROI_age_sex_site_M00_M03_v4labels_280rois_cat12_7.csv" #"df_ROI_age_sex_site_M00_M03_v4labels.csv"
RLINK_DATAFRAME_ALL_M00 = DATA_DIR + "df_ROI_age_sex_site_M00_v4labels_280rois_cat12_7.csv" #"df_ROI_age_sex_site_M00_v4labels.csv"
SELECTED_REGIONS_OF_INTEREST_RLINK = ['lHip_GM_Vol','rHip_GM_Vol','lAmy_GM_Vol', 'rAmy_GM_Vol'] #["Left Hippocampus_GM_Vol","Right Hippocampus_GM_Vol", "Left Amygdala_GM_Vol", "Right Amygdala_GM_Vol"] #
CV_SPLITS=ROOT+"03_classif_rois/stratified-5cv.json"
RLINK_DATAFRAME_280ROIS = "/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.7_roi/neuromorphometrics_cat12_vbm_roi.tsv"

# outputs
MODEL_DIR=ROOT+"models/normative_model/"
ZSCORES_PATH=ROOT+"reports/normative_modeling/"
# - roi_values has shape (n_subjects, n_rois)
# - covariates has shape (n_subjects, 2) for [age, sex]

# RLINK tests
def get_M0_M3():
    """
        returns two dataframes of ROIs at baseline (M0) and 3 months after li intake (M3) 
    """
    df_M0M3 = pd.read_csv(RLINK_DATAFRAME_M00_M03)
    df_M0 = pd.read_csv(RLINK_DATAFRAME_M00_M03) #pd.read_csv(RLINK_DATAFRAME_ALL_M00)
    df_M3 = df_M0M3[df_M0M3["session"] == "M03"].copy()
    df_M0 = df_M0M3[df_M0M3["session"]=="M00"].copy()
    df_M3 = df_M3.reset_index(drop=True)
    df_M0 = df_M0.reset_index(drop=True)

    df_M0["response"] = df_M0["response"].replace({"GR": 1, "PaR": 0, "NR": 0})
    df_M3["response"] = df_M3["response"].replace({"GR": 1, "PaR": 0, "NR": 0})

    return df_M0, df_M3

def fit_normative_model(rois_HC, residualize=False, evaluate=True, exclude_biobd_bsnip_HC=False, distinction_btw_hemispheres=True):
    # df_HC = pd.read_csv(OPENBHB_DATAFRAME)
    
    if exclude_biobd_bsnip_HC: df_HC = pd.read_csv(DF_PATH_HEALTHY_BRAINS_NO_BIOBD_BSNIP_HC)
    else: df_HC = pd.read_csv(DF_PATH_HEALTHY_BRAINS)
    if not distinction_btw_hemispheres:
        df_HC = get_mean_roi_left_right_hemispheres(df_HC)
        
    if 'participant_id' in df_HC.columns:
        df_HC = df_HC.drop(columns='participant_id')

    print("nb of rois :",len(rois_HC))
    
    df_train, df_test, train_idx, test_idxs  = stratified_split_balanced_age(df_HC, test_size=0.1) #stratified_split(df_HC, verbose=False)

    assert len(df_HC)==len(df_train)+len(df_test)
    y_tr =  df_train[rois_HC].values
    y_te =  df_test[rois_HC].values

    X_tr = df_train[['age', 'sex']]
    X_te = df_test[['age', 'sex']]
    print("Xtr , ytr",np.shape(X_tr), np.shape(y_tr))
    print("X_te , y_te",np.shape(X_te), np.shape(y_te))

    if residualize: # residualizing on site only
        # print("openbhb train set before residualization:\n", y_tr)
        df_all_cov = pd.concat([df_train[["age","sex","site"]], df_test[["age","sex","site"]]], axis=0, ignore_index=True)
        assert len(df_all_cov)==len(df_train)+len(df_test)

        residualizer = Residualizer(data=df_all_cov, formula_res="site", formula_full="site + sex + age")
        Zres = residualizer.get_design_mat(df_all_cov)
        residualizer.fit(y_tr, Zres[:len(df_train)])
        
        y_tr = residualizer.transform(y_tr, Zres[:len(df_train)])
        y_te = residualizer.transform(y_te, Zres[len(df_train):])
        # print("openbhb train set after residualization:\n",y_tr)

    model = NormativeBLR(warp=True, bsplines=True)
    model.fit(X_tr, y_tr)

    # healthy cohort tests
    if evaluate: 
        # Predict means
        roi_pred_tr = model.predict(X_tr)
        print("r2 on train set ", r2_score(y_tr, roi_pred_tr))

        roi_pred_te = model.predict(X_te)
        r2 = r2_score(y_te, roi_pred_te)
        print("r2 score on test set",r2)

        # Per-ROI r2 on test set (more informative)
        r2_per_roi = [r2_score(y_te[:, i], roi_pred_te[:, i]) 
                    for i in range(y_te.shape[1])]
        print(f"Per-ROI R² - Mean: {np.mean(r2_per_roi):.3f}, "
          f"Min: {np.min(r2_per_roi):.3f}, Max: {np.max(r2_per_roi):.3f}")

        # predict and get z-scores
        roi_pred, z_scores = model.predict(X_te, y_te, return_zscores=True)
        print("z_scores ",np.shape(z_scores), type(z_scores))
        print(f"\nZ-score statistics (should be ~N(0,1)):")
        print(f"Mean: {np.mean(z_scores):.3f} (should be close to 0)")
        print(f"Std: {np.std(z_scores):.3f} (should be close to 1)")

        # Per-ROI z-score check
        z_means = np.mean(z_scores, axis=0)
        z_stds = np.std(z_scores, axis=0)
        print(f"Per-ROI z-score means - range: [{z_means.min():.3f}, {z_means.max():.3f}]")
        print(f"Per-ROI z-score stds - range: [{z_stds.min():.3f}, {z_stds.max():.3f}]")
        
        df_yte = df_test[rois_HC]
        df_zscores_te = pd.DataFrame(z_scores, columns = rois_HC)

        # corr_matrix_te = df_yte.corr()
        # corr_matrix_te_zscores = df_zscores_te.corr()

        # cond_number_te = np.linalg.cond(corr_matrix_te.values)
        # fro_norm = np.linalg.norm(corr_matrix_te.values, ord='fro')
        # print(f"Frobenius norm: {fro_norm:.3f}")
        # print(f"Condition number: {cond_number_te:.2e}")

        # cond_number_te_zscores = np.linalg.cond(corr_matrix_te_zscores.values)
        # fro_norm = np.linalg.norm(corr_matrix_te_zscores.values, ord='fro')
        # print(f"Frobenius norm: {fro_norm:.3f}")
        # print(f"Condition number: {cond_number_te_zscores:.2e}")

    return model

"""

avec residualization sur site:
r2 on train set  0.319700273315677
r2 score on test set 0.3235918336401152
Per-ROI R² - Mean: 0.324, Min: -0.014, Max: 0.644
z_scores  (5649, 252) <class 'numpy.ndarray'>

Z-score statistics (should be ~N(0,1)):
Mean: -0.009 (should be ~0)
Std: 0.988 (should be ~1)
Per-ROI z-score means - range: [-0.272, 0.247]
Per-ROI z-score stds - range: [0.477, 1.082]

sans : 
r2 on train set  0.32588250110329264
r2 score on test set 0.32878945248819824
Per-ROI R² - Mean: 0.329, Min: -0.017, Max: 0.581
z_scores  (5649, 252) <class 'numpy.ndarray'>

Z-score statistics (should be ~N(0,1)):
Mean: 0.021 (should be close to 0)
Std: 0.982 (should be close to 1)
Per-ROI z-score means - range: [-0.033, 0.151]
Per-ROI z-score stds - range: [0.490, 1.029]

"""




def get_zscores(model, list_roi, df_M0, df_M3, residualize_on_site=False,distinction_btw_hemispheres=True):

    # responses
    roi_values_M0 = df_M0[list_roi].values
    roi_values_M3 = df_M3[list_roi].values

    # covariates
    X_rlink_M0 = df_M0[['age', 'sex']].copy()
    X_rlink_M3 = df_M3[['age', 'sex']].copy()
    # note : "sex" must be "female" or "male" (not 0 or 1)
    # making sure age is float
    X_rlink_M0['age'] = X_rlink_M0['age'].astype(float)
    X_rlink_M3['age'] = X_rlink_M3['age'].astype(float)

    if residualize_on_site:
        residualizer_M0 = Residualizer(data=df_M0[["age","sex","site"]], formula_res="site", formula_full="site + sex + age")
        residualizer_M3 = Residualizer(data=df_M3[["age","sex","site"]], formula_res="site", formula_full="site + sex + age")

        ZresM0 = residualizer_M0.get_design_mat(df_M0[["age","sex","site"]])
        ZresM3 = residualizer_M3.get_design_mat(df_M3[["age","sex","site"]])

        residualizer_M0.fit(roi_values_M0, ZresM0)
        residualizer_M3.fit(roi_values_M3, ZresM3)

        roi_values_M0 = residualizer_M0.transform(roi_values_M0, ZresM0)
        roi_values_M3 = residualizer_M3.transform(roi_values_M3, ZresM3)

    roi_pred_M0, z_scores_M0 = model.predict(X_rlink_M0, roi_values_M0, return_zscores=True)
    roi_pred_M3, z_scores_M3 = model.predict(X_rlink_M3, roi_values_M3, return_zscores=True)

    df_zscores_M0 = pd.DataFrame(z_scores_M0, columns = list_roi)
    df_zscores_M0.insert(0, 'participant_id', df_M0["participant_id"])  
    df_zscores_M0["response"]=df_M0["response"]

    df_zscores_M3 = pd.DataFrame(z_scores_M3, columns = list_roi)
    df_zscores_M3.insert(0, 'participant_id', df_M3["participant_id"])  
    df_zscores_M3["response"]=df_M3["response"]

    if distinction_btw_hemispheres:
        print("zscores M0 \n",df_zscores_M0[SELECTED_REGIONS_OF_INTEREST_RLINK])
        print("zscores M3 \n",df_zscores_M3[SELECTED_REGIONS_OF_INTEREST_RLINK])
    else: 
        print("zscores M0 \n",df_zscores_M0[["Hip_GM_Vol","Amy_GM_Vol","Pal_GM_Vol","Put_GM_Vol"]])
        print("zscores M3 \n",df_zscores_M3[["Hip_GM_Vol","Amy_GM_Vol","Pal_GM_Vol","Put_GM_Vol"]])
    
    return df_zscores_M0, df_zscores_M3


def plot_normative_roi_by_age(df, roi_name, normative_model, 
                                age_col='age', class_col="response", sex_col='sex',
                                figsize=(12, 8)):
    """
    Plot z-scores from normative model by age, with healthy normative curve.
    
    Parameters:
    -----------
    df_zscores : pandas.DataFrame
        DataFrame containing age, sex, class labels, and pre-computed z-scores
    roi_name : str
        Name of the ROI z-score column to plot
    normative_model : NormativeBLR
        Fitted normative model (used to plot healthy normative curve)
    """
    
    if roi_name not in df.columns:
        raise ValueError(f"ROI '{roi_name}' not found in columns")
    
    # Extract bipolar subject data
    ages = df[age_col].values
    zscores = df[roi_name].values
    classes = df[class_col].values
    
    # Separate by class
    class_0_mask = classes == 0
    class_1_mask = classes == 1
    
    ages_0 = ages[class_0_mask]
    ages_1 = ages[class_1_mask]
    zscores_0 = zscores[class_0_mask]
    zscores_1 = zscores[class_1_mask]
    
    # Create age range for plotting normative curve
    age_min, age_max = ages.min(), ages.max()
    age_range = np.linspace(age_min, age_max, 100)
    
    # Get normative predictions for both sexes
    roi_cols = [col for col in df.columns 
                if col not in [age_col, sex_col, class_col, 'participant_id']]
    roi_idx = roi_cols.index(roi_name.replace('_zscore', '')) if roi_name.replace('_zscore', '') in roi_cols else 0
    
    # Plot normative curves for both sexes
    fig, ax = plt.subplots(figsize=figsize)
    
    for sex_val in ["male", "female"]:
        X_norm = pd.DataFrame({
            'age': age_range,
            'sex': [sex_val] * len(age_range)
        })
        # X_norm = np.column_stack([age_range, np.full(len(age_range), sex_val)])
        predictions_norm = normative_model.predict(X_norm)
        
        sex_label = 'Female' if sex_val == 0 else 'Male'
        color = 'pink' if sex_val == 0 else 'lightblue'
        ax.plot(age_range, predictions_norm[:, roi_idx], 
                color=color, alpha=0.7, linewidth=2, 
                label=f'Healthy {sex_label} mean')
        
        # Add confidence bands (±1 std)
        std_val = normative_model.stds_[roi_idx]
        ax.fill_between(age_range, 
                       predictions_norm[:, roi_idx] - std_val,
                       predictions_norm[:, roi_idx] + std_val,
                       color=color, alpha=0.2)
    
    # Plot bipolar subject z-scores
    ax.scatter(ages_0, zscores_0, alpha=0.7, s=50, 
              color='blue', label=f'Bipolar Class 0 (n={len(ages_0)})')
    ax.scatter(ages_1, zscores_1, alpha=0.7, s=50, 
              color='red', label=f'Bipolar Class 1 (n={len(ages_1)})')
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    ax.axhline(y=1.96, color='gray', linestyle='--', alpha=0.7, label='±1.96σ')
    ax.axhline(y=-1.96, color='gray', linestyle='--', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Brain measure / Z-score', fontsize=12)
    ax.set_title(f'Normative Model: {roi_name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_normative_zscore_by_age(df_zscores, roi_name, 
                                 figsize=(12, 8)):
    """
    Plot z-scores from normative model by age, colored by class label.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing subjects with age, class labels, and ROI data
    roi_name : str
        Name of the ROI column to plot
    figsize : tuple
        Figure size (default: (12, 8))
    """
    
    # Check if ROI exists in dataframe
    if roi_name not in df_zscores.columns:
        raise ValueError(f"ROI '{roi_name}' not found in columns")
    
    # Extract data
    ages = df_zscores['age'].values
    zscores = df_zscores[roi_name].values
    classes = df_zscores["response"].values
    
    # Separate by class
    class_0_mask = classes == 0
    class_1_mask = classes == 1
    
    ages_0 = ages[class_0_mask]
    ages_1 = ages[class_1_mask]
    zscores_0 = zscores[class_0_mask]
    zscores_1 = zscores[class_1_mask]

    palette = sns.color_palette()
    fig, ax = plt.subplots(figsize=figsize)
    
    lowess_0 = lowess(zscores_0, ages_0, frac=0.3)
    ax.plot(lowess_0[:, 0], lowess_0[:, 1], color=palette[0], 
            linewidth=3, label=f'Non responders (n={len(ages_0)})')
    
    # Optional: Add scatter points with lower alpha
    ax.scatter(ages_0, zscores_0, alpha=0.3, s=30, color=palette[0])

    lowess_1 = lowess(zscores_1, ages_1, frac=0.3)
    ax.plot(lowess_1[:, 0], lowess_1[:, 1], color=palette[1], 
            linewidth=3, label=f'Responders (including partial responders) (n={len(ages_1)})')
    
    # Optional: Add scatter points with lower alpha
    ax.scatter(ages_1, zscores_1, alpha=0.3, s=30, color=palette[1])
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.9, linewidth=2.5)
    ax.axhline(y=1.96, color='gray', linestyle='--', alpha=0.7, label='±1.96σ')
    ax.axhline(y=-1.96, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Z-score', fontsize=12)
    ax.set_title(f'Deviation from normative "healthy" gray matter (Z-score = 0) in : {roi_name[:-len("_GM_Vol")]}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_normative_roi_by_age_using_zscores_no_dif_hemispheres(df_zscoresM0, roi_name, normative_model,
                             age_col='age', class_col="response", sex_col='sex',
                             figsize=(12, 12), lowess_frac=0.6):
    """
    Plot original ROI values converted from z-scores, with separate plots for male/female.
    
    Parameters:
    -----------
    df_zscores : pandas.DataFrame
        DataFrame containing age, sex, class labels
    roi_name : str
        Name of the ROI z-score column to plot
    normative_model : NormativeBLR
        Fitted normative model
    lowess_frac : float
        Fraction of data used for LOWESS smoothing
    """
    
    if roi_name not in df_zscoresM0.columns:
        raise ValueError(f"ROI '{roi_name}' not found in columns")
    
    # get ROI index
    roi_cols = [col for col in df_zscoresM0.columns 
                if col not in [age_col, sex_col, class_col, 'participant_id']]
    roi_idx = roi_cols.index(roi_name)
    
    # Normative predictions for M0 (can include more subjects)
    X_covariates_M0 = df_zscoresM0[[age_col, sex_col]]
    predictions_M0 = normative_model.predict(X_covariates_M0)
    residual_std = normative_model.stds_[roi_idx]
    
    # same thing for M0
    zscores_M0 = df_zscoresM0[roi_name].values
    ages_M0 = df_zscoresM0[age_col].values
    classes_M0 = df_zscoresM0[class_col].values
    sexes_M0 = df_zscoresM0[sex_col].values
    original_volumes_M0 = (zscores_M0 * residual_std) + predictions_M0[:, roi_idx]

    # Get seaborn colors
    colors = sns.color_palette()
    blue_color = colors[0]    # Non-responders
    orange_color = colors[1]  # Responders
    
    # Create subplots: Female on top (sex=1), Male below (sex=0)
    fig, (ax_female, ax_male) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Age range for normative curves
    age_min, age_max = ages_M0.min(), ages_M0.max()
    age_range = np.linspace(age_min, age_max, 100)
    
    for sex_val, ax, sex_title in [("female", ax_female, 'Female'), ("male", ax_male, 'Male')]:
        # filter data by sex
        sex_mask_M0 = sexes_M0 == sex_val
        X_norm = pd.DataFrame({
            'age': age_range,
            'sex': [sex_val] * len(age_range)
        })

        # ---- HC ----
        predictions_norm = normative_model.predict(X_norm)
        # print("sex_val ",sex_val)
        # print("X_norm ",X_norm)
        # print("predictions_norm[:, roi_idx] ",predictions_norm[:, roi_idx])
        
        # Plot normative curve in grey
        ax.plot(age_range, predictions_norm[:, roi_idx], 
                color='grey', alpha=0.8, linewidth=2) # ,label='Healthy mean')
        
        # Add confidence bands (±1 std) in grey
        ax.fill_between(age_range, 
                       predictions_norm[:, roi_idx] - residual_std,
                       predictions_norm[:, roi_idx] + residual_std,
                       color='grey', alpha=0.2)#, label='±1 SD')


        # Get rlink M0 data for this sex
        sex_ages_M0 = ages_M0[sex_mask_M0]
        sex_volumes_M0 = original_volumes_M0[sex_mask_M0]
        sex_classes_M0 = classes_M0[sex_mask_M0]
        
        # Separate by rlink response class within this sex
        class_0_mask_sex = sex_classes_M0 == 0
        class_1_mask_sex = sex_classes_M0 == 1
        
        ages_0_sex = sex_ages_M0[class_0_mask_sex]
        ages_1_sex = sex_ages_M0[class_1_mask_sex]
        volumes_0_sex = sex_volumes_M0[class_0_mask_sex]
        volumes_1_sex = sex_volumes_M0[class_1_mask_sex]
        
        # Plot scatter points
        if len(ages_0_sex) > 0:
            ax.scatter(ages_0_sex, volumes_0_sex, alpha=0.6, s=40,
                      color=blue_color)#, label=f'Non or partial responders') #(n={len(ages_0_sex)})
        
        if len(ages_1_sex) > 0:
            ax.scatter(ages_1_sex, volumes_1_sex, alpha=0.6, s=40,
                      color=orange_color)#, label=f'Good responders') #(n={len(ages_1_sex)})
        
        # Add LOWESS curves
        if len(ages_0_sex) > 3: # condition to make sure lowess estimates are ok for plotting
            lowess_0 = lowess(volumes_0_sex, ages_0_sex, frac=lowess_frac)
            ax.plot(lowess_0[:, 0], lowess_0[:, 1], color=blue_color, 
                    linewidth=3, alpha=0.8)
        
        if len(ages_1_sex) > 3:
            lowess_1 = lowess(volumes_1_sex, ages_1_sex, frac=lowess_frac)
            ax.plot(lowess_1[:, 0], lowess_1[:, 1], color=orange_color, 
                    linewidth=3, alpha=0.8)
            
        # Formatting for each subplot
        ax.set_ylabel('Gray Matter Volume', fontsize=12)
        ax.set_title(f'{sex_title}', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle=':', linewidth=1, alpha=0.4)
    
    # Set x-label only for bottom plot
    ax_male.set_xlabel('Age (years)', fontsize=12)

    # Overall title
    fig.suptitle(f'Normative Model: {roi_name[:-len("_GM_Vol")]}', fontsize=16, y=0.98)
    
    plt.tight_layout()

    plt.show()
    return fig, (ax_female, ax_male)

def plot_normative_roi_by_age_using_zscores(df_zscoresM0, df_zscoresM3, roi_name, normative_model,
                             age_col='age', class_col="response", sex_col='sex',
                             figsize=(12, 12), lowess_frac=0.6):
    """
    Plot original ROI values converted from z-scores, with separate plots for male/female.
    
    Parameters:
    -----------
    df_zscores : pandas.DataFrame
        DataFrame containing age, sex, class labels
    roi_name : str
        Name of the ROI z-score column to plot
    normative_model : NormativeBLR
        Fitted normative model
    lowess_frac : float
        Fraction of data used for LOWESS smoothing
    """
    
    if roi_name not in df_zscoresM3.columns or roi_name not in df_zscoresM0.columns:
        raise ValueError(f"ROI '{roi_name}' not found in columns")
    
    # get ROI index
    roi_cols = [col for col in df_zscoresM3.columns 
                if col not in [age_col, sex_col, class_col, 'participant_id']]
    roi_idx = roi_cols.index(roi_name)
    
    # get model predictions for M3 subjects
    X_covariates_M3 = df_zscoresM3[[age_col, sex_col]]
    predictions_M3 = normative_model.predict(X_covariates_M3)
    residual_std = normative_model.stds_[roi_idx]

    # Normative predictions for M0 (can include more subjects)
    X_covariates_M0 = df_zscoresM0[[age_col, sex_col]]
    predictions_M0 = normative_model.predict(X_covariates_M0)
    
    # convert z-scores for M3 back to original GM volumes
    zscoresM3 = df_zscoresM3[roi_name].values
    original_volumesM3 = (zscoresM3 * residual_std) + predictions_M3[:, roi_idx]
    
    # same thing for M0
    zscores_M0 = df_zscoresM0[roi_name].values
    ages_M0 = df_zscoresM0[age_col].values
    classes_M0 = df_zscoresM0[class_col].values
    sexes_M0 = df_zscoresM0[sex_col].values
    original_volumes_M0 = (zscores_M0 * residual_std) + predictions_M0[:, roi_idx]


    # Extract data
    ages_M3 = df_zscoresM3[age_col].values
    classes_M3 = df_zscoresM3[class_col].values
    sexes_M3 = df_zscoresM3[sex_col].values
    
    # Get seaborn colors
    colors = sns.color_palette()
    blue_color = colors[0]    # Non-responders
    orange_color = colors[1]  # Responders
    
    # Create subplots: Female on top (sex=1), Male below (sex=0)
    fig, (ax_female, ax_male) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Age range for normative curves
    age_min, age_max = ages_M0.min(), ages_M0.max()
    age_range = np.linspace(age_min, age_max, 100)
    
    for sex_val, ax, sex_title in [("female", ax_female, 'Female'), ("male", ax_male, 'Male')]:
        # filter data by sex
        sex_mask_M3 = sexes_M3 == sex_val
        X_norm = pd.DataFrame({
            'age': age_range,
            'sex': [sex_val] * len(age_range)
        })

        # ---- HC ----
        predictions_norm = normative_model.predict(X_norm)
        # print("sex_val ",sex_val)
        # print("X_norm ",X_norm)
        # print("predictions_norm[:, roi_idx] ",predictions_norm[:, roi_idx])
        
        # Plot normative curve in grey
        ax.plot(age_range, predictions_norm[:, roi_idx], 
                color='grey', alpha=0.8, linewidth=2) # ,label='Healthy mean')
        
        # Add confidence bands (±1 std) in grey
        ax.fill_between(age_range, 
                       predictions_norm[:, roi_idx] - residual_std,
                       predictions_norm[:, roi_idx] + residual_std,
                       color='grey', alpha=0.2)#, label='±1 SD')

        # ---- M3 ----

        # Get rlink M3 data for this sex
        sex_ages_M3 = ages_M3[sex_mask_M3]
        sex_volumes_M3 = original_volumesM3[sex_mask_M3]
        sex_classes_M3 = classes_M3[sex_mask_M3]
        
        # Separate by rlink response class within this sex
        class_0_mask_sex = sex_classes_M3 == 0
        class_1_mask_sex = sex_classes_M3 == 1
        
        ages_0_sex = sex_ages_M3[class_0_mask_sex]
        ages_1_sex = sex_ages_M3[class_1_mask_sex]
        volumes_0_sex = sex_volumes_M3[class_0_mask_sex]
        volumes_1_sex = sex_volumes_M3[class_1_mask_sex]
        
        # Plot scatter points
        if len(ages_0_sex) > 0:
            ax.scatter(ages_0_sex, volumes_0_sex, alpha=0.6, s=40,
                      color=blue_color)#, label=f'Non or partial responders') #(n={len(ages_0_sex)})
        
        if len(ages_1_sex) > 0:
            ax.scatter(ages_1_sex, volumes_1_sex, alpha=0.6, s=40,
                      color=orange_color)#, label=f'Good responders') #(n={len(ages_1_sex)})
        
        # Add LOWESS curves
        if len(ages_0_sex) > 3: # condition to make sure lowess estimates are ok for plotting
            lowess_0 = lowess(volumes_0_sex, ages_0_sex, frac=lowess_frac)
            ax.plot(lowess_0[:, 0], lowess_0[:, 1], color=blue_color, linestyle='--',
                    linewidth=3, alpha=0.8)
        
        if len(ages_1_sex) > 3:
            lowess_1 = lowess(volumes_1_sex, ages_1_sex, frac=lowess_frac)
            ax.plot(lowess_1[:, 0], lowess_1[:, 1], color=orange_color, linestyle='--',
                    linewidth=3, alpha=0.8)
            
        # ---- M0 ----
        sex_mask_M0 = sexes_M0 == sex_val
        sex_ages_M0 = ages_M0[sex_mask_M0]
        sex_volumes_M0 = original_volumes_M0[sex_mask_M0]
        sex_classes_M0 = classes_M0[sex_mask_M0]

        class_0_M0 = sex_classes_M0 == 0
        class_1_M0 = sex_classes_M0 == 1
        if np.sum(class_0_M0) > 3:
            lowess_0_M0 = lowess(sex_volumes_M0[class_0_M0],sex_ages_M0[class_0_M0],frac=lowess_frac)
            ax.plot(lowess_0_M0[:, 0], lowess_0_M0[:, 1], color=blue_color, linewidth=3, alpha=0.4)

        if np.sum(class_1_M0) > 3:
            lowess_1_M0 = lowess(sex_volumes_M0[class_1_M0],sex_ages_M0[class_1_M0],frac=lowess_frac)
            ax.plot(lowess_1_M0[:, 0], lowess_1_M0[:, 1],color=orange_color,linewidth=3, alpha=0.4)
        
        # Formatting for each subplot
        ax.set_ylabel('Gray Matter Volume', fontsize=12)
        ax.set_title(f'{sex_title}', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle=':', linewidth=1, alpha=0.4)
    
    # Set x-label only for bottom plot
    ax_male.set_xlabel('Age (years)', fontsize=12)
    # legend_elements = [
    #     # Normative
    #     Line2D([0], [0], color='grey', lw=2, label='Healthy mean'),
    #     Patch(facecolor='grey', alpha=0.2, label='±1 SD'),

    #     # Responders
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor=blue_color,
    #         markersize=8, label='Non or partial responders'),
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor=orange_color,
    #         markersize=8, label='Good responders'),

    #     # Timepoint encoding
    #     Line2D([0], [0], color='black', lw=3, linestyle='-',
    #         label='After lithium (M3)'),
    #     Line2D([0], [0], color='black', lw=3, linestyle='--',
    #         label='Before lithium (M0)')
    # ]
    # fig.legend(
    #     handles=legend_elements,
    #     loc='center right',
    #     bbox_to_anchor=(0.98, 0.5),   # inside image, right side of plot
    #     frameon=True,
    #     fontsize=13
    # )

    # Overall title
    fig.suptitle(f'Normative Model: {roi_name[:-len("_GM_Vol")]}', fontsize=16, y=0.98)
    
    plt.tight_layout()

    plt.show()
    return fig, (ax_female, ax_male)

def plot_age_distrib_resp_or_noresp(df, y):
    """
    generates bar plot of age distribution for responders (y=1) or non responders (y=0)
    bars are colored by sex (male = 0, female = 1)
    """
    # Filter data where y == 0
    df_filtered = df[df["response"] == y]

    # Create bar plot of age distribution colored by age
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_filtered, x='age', hue='sex')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title(f'Age Distribution for y={y}, colored by Sex')
    plt.legend(title='Sex', labels=['Male (0)', 'Female (1)'])
    plt.xticks(rotation=45)
    plt.show()


def classification_from_zscores(model, scale=False, residualize_on_site=False):
    with open(CV_SPLITS, "r") as f:
        splits = json.load(f)

    # _, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()
    df_M0, df_M3 = get_M0_M3()
    list_roi_rlink = [r for r in list(df_M0.columns) if r.endswith("_CSF_Vol") or r.endswith("_GM_Vol")]
    list_roi_rlinkM3= [r for r in list(df_M3.columns) if r.endswith("_CSF_Vol") or r.endswith("_GM_Vol")]
    

    assert list_roi_rlink==list_roi_rlinkM3
    quit()

    df, _ = get_zscores(model, list_roi_rlink, df_M0, df_M3) 
    # warning : in get_zscores: do NOT apply residualization to all zscores at once (data leakage for classification)

    assert max(x for outer in splits for inner in outer for x in inner) == 116
    X = df.drop(columns=["response","participant_id"]).values
    y = df["response"].values
    # Define model
    clf = lm.LogisticRegression(fit_intercept=True, class_weight='balanced')
    # clf = GridSearchCV(
    #         lm.LogisticRegression(class_weight="balanced", fit_intercept=False),
    #         param_grid={"C": [0.1, 1.0, 10.0]},
    #         cv=3,            
    #         scoring="roc_auc",  
    #         n_jobs=-1
    #     )

    scaler = StandardScaler()
    scores = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if residualize_on_site:
            residualizer = Residualizer(data=df_M0[["age","sex","site"]], formula_res="site", formula_full="site + sex + age")
            Zres = residualizer.get_design_mat(df_M0[["age","sex","site"]])
            assert len(Zres)==len(X)
            residualizer.fit(X_train, Zres[train_idx])
            X_train = residualizer.transform(X_train, Zres[train_idx])
            X_test = residualizer.transform(X_test, Zres[test_idx])

        if scale:
            # Scale features
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Fit model
        clf.fit(X_train, y_train)

        # Predictions
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        scores.append({"fold": fold, "accuracy": acc, "roc_auc": auc})

    # Convert results to DataFrame
    df_scores = pd.DataFrame(scores)
    print(df_scores)
    print("Mean AUC:", df_scores["roc_auc"].mean())

def get_roi_diff_percentages(df0, df3, list_roi):
    """
    gives percentages of participants with "normalized" zscores after lithium intake 
    for good responders and non/partial responders separately

    outputs a dict where key 0 is NR/PaR and key 1 is GR

    df0 : dataframe with M0 values
    df3 : dataframe with M3 values
    list_roi : list of rois
    """
    assert df3["participant_id"].is_unique

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

    results = {}

    for label in [0, 1]:
        # Filter rows where y == label
        mask = df0_common["response"] == label
        d0 = df0_common.loc[mask, list_roi].abs()
        d3 = df3_common.loc[mask, list_roi].abs()

        # Compute per-ROI % of participants where |df3| < |df0|
        # --> equivalent to a closening to the "norm" after Li intake
        percentages = (d3 < d0).sum(axis=0) / len(d0) * 100
        results[label] = round(percentages,2)

    return results  # dict: {0: Series, 1: Series}

def get_mean_roi_left_right_hemispheres(df):
    left_cols = [c for c in df.columns if c.startswith("l") and c.endswith("_Vol")]
    cols_not_roi = [c for c in df.columns if not c.endswith("_Vol")]

    new_df = df[cols_not_roi].copy()
    averaged_rois = {}
    for lcol in left_cols:
        base_name = lcol[1:]
        rcol = "r" + base_name
        
        if rcol in df.columns:
            averaged_rois[base_name] = df[[lcol, rcol]].mean(axis=1)
    
    new_df = pd.concat([new_df, pd.DataFrame(averaged_rois, index=df.index)], axis=1)
    return new_df

def save_zscores(residualize_on_site=False, exclude_biobd_bsnip_HC=False, distinction_btw_hemispheres=True):
    """
    1. fit normative model if it doesn't already exist
    2. save df of zscores (if it doesn't already exist) for either
        A. a normative model trained on Healthy Controls ROI residualized (on site),
            and zscores estimated from RLINK ROI residualized (on site)
        B. a normative model trained on Healthy Controls ROI and zscores estimated from RLINK ROI, 
            neither residualized on site
    """
    exclude_biobd_bsnip_HC_str = "_exclude_biobd_bsnip_HC" if exclude_biobd_bsnip_HC else ""
    distinction_btw_hemispheres_str = "_mean_btw_hemispheres" if not distinction_btw_hemispheres else ""
    if residualize_on_site: modelname = f"normative_BLR_BigHC_site_residualized_janv2026_280rois{exclude_biobd_bsnip_HC_str}{distinction_btw_hemispheres_str}"
    else : modelname = f"normative_BLR_BigHC_janv2026_280rois{exclude_biobd_bsnip_HC_str}{distinction_btw_hemispheres_str}" 

    model_path = MODEL_DIR+modelname+".pkl"

    # _, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()
    if exclude_biobd_bsnip_HC: df_HC = pd.read_csv(DF_PATH_HEALTHY_BRAINS_NO_BIOBD_BSNIP_HC)
    else: df_HC = pd.read_csv(DF_PATH_HEALTHY_BRAINS)

    if not distinction_btw_hemispheres:
        df_HC = get_mean_roi_left_right_hemispheres(df_HC)

    rois_HC = list(df_HC.columns)
    rois_HC = [ r for r in rois_HC if r not in ["participant_id","age","sex","site","cohort"]]

    if distinction_btw_hemispheres: assert len(rois_HC)==280
    else: assert len(rois_HC)==138

    if not os.path.exists(model_path):
        model = fit_normative_model(rois_HC, residualize=residualize_on_site, exclude_biobd_bsnip_HC=exclude_biobd_bsnip_HC, distinction_btw_hemispheres=distinction_btw_hemispheres)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    df_M0, df_M3 = get_M0_M3()

    if not distinction_btw_hemispheres:
        df_M0 = get_mean_roi_left_right_hemispheres(df_M0)
        df_M3 = get_mean_roi_left_right_hemispheres(df_M3)

    if distinction_btw_hemispheres:
        assert len(rois_HC) == 280
        assert len(list(df_M0[rois_HC].columns))==280
        assert len(list(df_M3[rois_HC].columns))==280
    else: 
        assert len(rois_HC) == 138
        assert len(list(df_M0[rois_HC].columns))==138
        assert len(list(df_M3[rois_HC].columns))==138

    print(f"N= {len(df_M0)} M0 RLink participants , and N= {len(df_M3)} RLink M3 participants. ")

    df_zscores_M0, df_zscores_M3 = get_zscores(model, rois_HC, df_M0, df_M3, residualize_on_site=residualize_on_site, distinction_btw_hemispheres=distinction_btw_hemispheres)
    # classification_from_zscores(model, scale=True, residualize_on_site=True)

    # plot_age_distrib_resp_or_noresp(df_M0, y=1)
    df_zscores_M0_with_age_sex_site = pd.merge(df_zscores_M0, df_M0[["participant_id","age","sex","site"]],on="participant_id", how="inner")
    df_zscores_M3_with_age_sex_site  = pd.merge(df_zscores_M3, df_M0[["participant_id","age","sex","site"]],on="participant_id", how="inner")
    
    # save dataframes of zscores to csv
    pathzscoresM0 = ZSCORES_PATH+f"df_zscores_M0_with_age_sex_site_{modelname}_cat12_7_280ROIs{exclude_biobd_bsnip_HC_str}{distinction_btw_hemispheres_str}.csv"
    pathzscoresM3 = ZSCORES_PATH+f"df_zscores_M3_with_age_sex_site_{modelname}_cat12_7_280ROIs{exclude_biobd_bsnip_HC_str}{distinction_btw_hemispheres_str}.csv"
    if not os.path.exists(pathzscoresM0): df_zscores_M0_with_age_sex_site.to_csv(pathzscoresM0, index=False)
    if not os.path.exists(pathzscoresM3): df_zscores_M3_with_age_sex_site.to_csv(pathzscoresM3, index=False)

    # for i in range(4):
    #     plot_normative_zscore_by_age(df_zscores_M0_with_age_sex_site, SELECTED_REGIONS_OF_INTEREST_RLINK[i])
    # print("df_M0 \n",df_M0[SELECTED_REGIONS_OF_INTEREST_RLINK])
    if distinction_btw_hemispheres: 
        print("M3 GR zscores :\n", df_zscores_M3[df_zscores_M3["response"]==1][SELECTED_REGIONS_OF_INTEREST_RLINK+ ["rPal_GM_Vol","lPut_GM_Vol","lPal_GM_Vol","rPut_GM_Vol"]].mean())
        print("M3 NR/PaR zscores :\n",df_zscores_M3[df_zscores_M3["response"]==0][SELECTED_REGIONS_OF_INTEREST_RLINK+ ["rPal_GM_Vol","lPut_GM_Vol","lPal_GM_Vol","rPut_GM_Vol"]].mean())

        print("M0 GR zscores :\n", df_zscores_M0[df_zscores_M0["response"]==1][SELECTED_REGIONS_OF_INTEREST_RLINK+ ["rPal_GM_Vol","lPut_GM_Vol","lPal_GM_Vol","rPut_GM_Vol"]].mean())
        print("M0 NR/PaR zscores :\n",df_zscores_M0[df_zscores_M0["response"]==0][SELECTED_REGIONS_OF_INTEREST_RLINK+ ["rPal_GM_Vol","lPut_GM_Vol","lPal_GM_Vol","rPut_GM_Vol"]].mean())
    else:
        print("M3 GR zscores :\n", df_zscores_M3[df_zscores_M3["response"]==1][["Hip_GM_Vol","Amy_GM_Vol","Pal_GM_Vol","Put_GM_Vol"]].mean())
        print("M3 NR/PaR zscores :\n",df_zscores_M3[df_zscores_M3["response"]==0][["Hip_GM_Vol","Amy_GM_Vol","Pal_GM_Vol","Put_GM_Vol"]].mean())

        print("M0 GR zscores :\n", df_zscores_M0[df_zscores_M0["response"]==1][["Hip_GM_Vol","Amy_GM_Vol","Pal_GM_Vol","Put_GM_Vol"]].mean())
        print("M0 NR/PaR zscores :\n",df_zscores_M0[df_zscores_M0["response"]==0][["Hip_GM_Vol","Amy_GM_Vol","Pal_GM_Vol","Put_GM_Vol"]].mean())

    return df_zscores_M0_with_age_sex_site, df_zscores_M3_with_age_sex_site, model

def zscores_m3minusm0(residualize_on_site=False):
    """
    Saves a df of zscores M3 - zscores M0 ROI measures
        save_m3_minus_m0_df (bool) : 
            if True, save df of differences between zscores of m3 and zscores of m0 to csv. if False, don't.
        residualize_on_site: when the NM has been trained on ROI residualized on site, and the zscores have been estimated
            from ROI residualized on site as well (warning : the sites are different when estimating the NM on healthy controls
            from the sites of the dataset used to estimate zscores)
    """
    if residualize_on_site:
        modelname = "normative_BLR_BigHC_site_residualized"
        new_zscores_df_file = f"df_zscores_M3minusM0_with_age_sex_site_res_on_site_{modelname}.csv"

    else : 
        modelname = "normative_BLR_BigHC_nov2025" 
        new_zscores_df_file = f"df_zscores_M3minusM0_with_age_sex_site_{modelname}.csv"

    _, list_roi_rlink = get_lists_roi_in_both_openBHB_and_rlink()

    path_df_M3minusM0 = ZSCORES_PATH+new_zscores_df_file

    if not os.path.exists(path_df_M3minusM0):
        df_ROI_age_sex_site_M0 = pd.read_csv(ZSCORES_PATH+f"df_zscores_M0_with_age_sex_site_{modelname}.csv")
        df_ROI_age_sex_site_M3 = pd.read_csv(ZSCORES_PATH+f"df_zscores_M3_with_age_sex_site_{modelname}.csv")
        print(df_ROI_age_sex_site_M3)
        print(df_ROI_age_sex_site_M0)
        # Merge M03 and M00 on participant_id
        merged = pd.merge(df_ROI_age_sex_site_M3, df_ROI_age_sex_site_M0, on="participant_id", suffixes=("_M03", "_M00"))

        list_ = ["response","age","sex","site"]
        for l in list_:
            assert (merged[l+'_M00'] == merged[l+'_M03']).all(), " issue with "+l+" between same subjects at M00 and M03"

        columns_M03 = [col for col in merged.columns if col.endswith("_M03")]
        columns_M00 = [col for col in merged.columns if col.endswith("_M00")]
        common_columns = [col[:-4] for col in columns_M03 if col[:-4] + "_M00" in columns_M00]
        print(len(list_roi_rlink))
        print("common_columns ", len(common_columns)) # == 273 since there are 268 ROIs (134 GM and 134 CSF) and response (label), age, sex, site, session
        print(merged)

        # creating a df for differences of M03-M00
        differences_df = pd.DataFrame({
            col: merged[col + "_M03"] - merged[col + "_M00"] for col in common_columns if not col in \
                ["participant_id", "age","sex","site","response","session"]
        })
        differences_df.insert(0, 'participant_id', merged['participant_id'])
        differences_df["response"] = merged["response_M00"]
        differences_df["age"] = merged["age_M00"]
        differences_df["sex"] = merged["sex_M00"]
        differences_df["site"] = merged["site_M00"]

        print(differences_df)
        differences_df.to_csv(path_df_M3minusM0,index=False)  

    else : differences_df = pd.read_csv(path_df_M3minusM0)

    print("differences grouped by label\n",differences_df[list_roi_rlink+["response"]].groupby("response").median())
    differences_df = differences_df.drop("participant_id", axis=1)
    print(differences_df)

def extreme_roi_stats(df, roi_cols, zthr=2):
    """
    For each ROI:
      - count values with |z| >= zthr
      - compute mean and median

    Returns a sorted DataFrame
    """
    rows = []

    for roi in roi_cols:
        z = df[roi].values

        n_extreme = np.sum(np.abs(z) >= zthr)
        mean_z = np.nanmean(z)
        median_z = np.nanmedian(z)

        rows.append({
            "ROI": roi,
            "n_|z|>=2": n_extreme,
            "mean_z": mean_z,
            "median_z": median_z
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("n_|z|>=2", ascending=False)

    return out


def print_demographics(df):
    for site, group in list(df.groupby("cohort")) + [("Total", df)]:
        n = group["participant_id"].nunique()
        age_mean, age_std = group["age"].mean(), group["age"].std()
        pct_female = (group["sex"] == "female").mean() * 100
        print(f"{site}: N={n}, age={age_mean:.1f}±{age_std:.1f}, female={pct_female:.1f}%")


def plot_age_distribution(df, name_df=""):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 5))
    
    for sex, color in [("female", "#E87D8A"), ("male", "#5B8DB8")]:
        subset = df[df["sex"] == sex]["age"]
        sns.kdeplot(
            subset,
            ax=ax,
            color=color,
            fill=True,
            alpha=0.4,
            linewidth=2,
            label=sex.capitalize(),
            common_norm=False  # <-- important if group sizes differ
        )

    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_title("Age distribution by sex", fontsize=23)
    ax.legend(title=None, fontsize=16)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"age_distribution_by_sex_{name_df}.png", dpi=150)
    plt.show()

def summarize_dataframe(df):
    summary = {}

    # Participants
    summary["n_participants"] = df["participant_id"].nunique()

    # Age
    summary["age_mean"] = df["age"].mean()
    summary["age_std"] = df["age"].std()

    # % Female
    if df["sex"].dtype == object:
        is_female = df["sex"].str.lower() == "female"
    else:
        is_female = df["sex"] == 0

    summary["pct_female"] = 100 * is_female.mean()

    # # for BIOBD BSNIP
    # if df["dx"].dtype == object:
    #     is_BD = df["dx"] == 1
    # else: 
    #     is_BD = df["dx"] == 0

    # summary["pct_BD"] = 100 * is_BD.mean()

    # for SCZ
    if df["dx"].dtype == object:
        is_SCZ = df["dx"] == 1
    else: 
        is_SCZ = df["dx"] == 0

    summary["pct_SCZ"] = 100 * is_SCZ.mean()

    # for RLink
    # % GR
    # if df["response"].dtype == object:
    #     is_GR = df["response"] == "GR"
    # else:
    #     is_GR = df["response"] == "NR/PaR"

    # summary["pct_GR"] = 100 * is_GR.mean()

    # Sites
    summary["n_sites"] = df["site"].nunique()

    # # Sites per cohort
    # sites_cohort = df.groupby("cohort")["site"].nunique()
    # for cohort, n_sites in sites_cohort.items():
    #     summary[f"sites_in_{cohort}"] = n_sites

    return pd.Series(summary)


def main():

    # print_demographics(df_HC)
    plot_age_distribution(DF_PATH_HEALTHY_BRAINS)

    quit()

    # rlink_df = pd.read_csv(RLINK_DATAFRAME_280ROIS , sep="\t")
    # save_zscores(False)
    # zscores_m3minusm0(residualize_on_site=True)
    # zscores_m3minusm0(residualize_on_site=False)
    
    # modelname="normative_BLR_BigHC_janv2026_280rois"
    # pathzscoresM0 = ZSCORES_PATH+"df_zscores_M0_with_age_sex_site_"+modelname+"_cat12_7_280ROIs.csv"
    # pathzscoresM3 = ZSCORES_PATH+"df_zscores_M3_with_age_sex_site_"+modelname+"_cat12_7_280ROIs.csv"
    # if os.path.exists(pathzscoresM0): df_zscores_M0_with_age_sex_site = pd.read_csv(pathzscoresM0)
    # if os.path.exists(pathzscoresM3): df_zscores_M3_with_age_sex_site = pd.read_csv(pathzscoresM3)


    df_zscores_M0_with_age_sex_site, df_zscores_M3_with_age_sex_site, model = save_zscores(residualize_on_site=False, exclude_biobd_bsnip_HC=False, distinction_btw_hemispheres=True)

    df_zscores_M0_with_age_sex_site_GR = df_zscores_M0_with_age_sex_site[df_zscores_M0_with_age_sex_site["response"]==1]
    df_zscores_M0_with_age_sex_site_ParNR = df_zscores_M0_with_age_sex_site[df_zscores_M0_with_age_sex_site["response"]==0]
    # df_HC = pd.read_csv(DF_PATH_HEALTHY_BRAINS)
    # rois_HC = list(df_HC.columns)
    # rois_HC= [ r for r in rois_HC if r not in ["participant_id","age","sex","site","cohort"]]
    # stats_GR = extreme_roi_stats(df_zscores_M0_with_age_sex_site_GR, rois_HC, zthr=2)
    # stats_ParNR = extreme_roi_stats(df_zscores_M0_with_age_sex_site_ParNR, rois_HC, zthr=2)
    # print(stats_GR.head(10))
    # print(stats_ParNR.head(10))
    # quit()

    # for roi in SELECTED_REGIONS_OF_INTEREST_RLINK:
    #     print(roi)
    #     results = get_roi_diff_percentages(df_zscores_M0_with_age_sex_site, df_zscores_M3_with_age_sex_site, [roi])
    #     print(results)
    


    for roi in ["lThaPro_GM_Vol"]: #"Pal_GM_Vol","Put_GM_Vol"]:#,"Hip_GM_Vol","Amy_GM_Vol"]:
        # plot_normative_roi_by_age_using_zscores_no_dif_hemispheres(df_zscores_M0_with_age_sex_site, roi, model)
        plot_normative_roi_by_age_using_zscores(df_zscores_M0_with_age_sex_site,df_zscores_M3_with_age_sex_site, roi, model)
    
    # quit()

    # for roi in ["rPal_GM_Vol","lPut_GM_Vol","lPal_GM_Vol","rPut_GM_Vol"]:#"lThaPro_GM_Vol","rMedPrcGy_GM_Vol", "lPrcGy_GM_Vol"]:
    #     print(roi)
    #     results = get_roi_diff_percentages(df_zscores_M0_with_age_sex_site, df_zscores_M3_with_age_sex_site, [roi])
    #     print(results)
    #     plot_normative_roi_by_age_using_zscores(df_zscores_M0_with_age_sex_site, df_zscores_M3_with_age_sex_site, roi, model)

    quit()
    """
    GR
                    ROI  n_|z|>=2    mean_z  median_z
    92       lOccPo_GM_Vol         9 -0.987021 -1.160139
    201     rFroPo_CSF_Vol         8  1.282089  1.441445
    222  lMedOrbGy_CSF_Vol         7  1.312121  1.464842
    74    lMidCinGy_GM_Vol         7 -0.851610 -0.899582
    212  lLatOrbGy_CSF_Vol         7  1.013908  0.834208
    196       lEnt_CSF_Vol         7  1.355405  1.352193
    253  rPosOrbGy_CSF_Vol         7  1.133939  1.379307
    197       rEnt_CSF_Vol         7  1.215915  1.114415
    204     lRecGy_CSF_Vol         6  0.929359  0.758914
    223  rMedOrbGy_CSF_Vol         6  1.126210  1.051744

    PaR/NR
                    ROI  n_|z|>=2    mean_z  median_z
    187  rAntOrbGy_CSF_Vol        21  1.508238  1.221645
    223  rMedOrbGy_CSF_Vol        20  1.309908  1.154670
    222  lMedOrbGy_CSF_Vol        18  1.483486  1.294154
    253  rPosOrbGy_CSF_Vol        16  1.113001  0.969741
    252  lPosOrbGy_CSF_Vol        14  1.254501  1.309570
    194       lCun_CSF_Vol        13  0.769084  0.667397
    57         rEnt_GM_Vol        13  0.953976  0.811158
    213  rLatOrbGy_CSF_Vol        13  0.933355  0.884284
    56         lEnt_GM_Vol        12  0.839409  0.765459
    273  rSupTemGy_CSF_Vol        11  0.602286  0.901951
    """

    for i in range(4):
        plot_normative_roi_by_age_using_zscores(df_zscores_M0_with_age_sex_site, df_zscores_M3_with_age_sex_site, SELECTED_REGIONS_OF_INTEREST_RLINK[i], model)
    
    # X_arr_res, residualizer_estimator, residualization_formula = \
    #             get_residualizer(df_M0, df_M0[list_roi_rlink].values, residualization_columns=["site"])
    # print("df_M0 before res :",df_M0[list_roi_rlink])
    # df_M0[list_roi_rlink] = residualizer_estimator.fit_transform(X_arr_res)
    # print("df_M0 after res :",df_M0[list_roi_rlink])

  
    # # print("correlation matrix Frobenius norm residualized M0 roi "\
    # #         ,np.linalg.norm(df_roi_M0_res.corr().values, ord='fro'))
    # plot_normative_roi_by_age(df_M0, SELECTED_REGIONS_OF_INTEREST_RLINK[1], model, 
    #                             age_col='age', class_col="response", sex_col='sex',
    #                             figsize=(12, 8))
    
    # results = get_roi_diff_percentages(df_zscores_M0, df_zscores_M3, list_roi_rlink)
    


    

if __name__ == "__main__":
    main()

"""

results of classification_from_zscores(df_zscores_M0, scale=True)

with normative_BLR_BigHC
classification from zscores with standard scaling:
    fold  accuracy   roc_auc
0     0  0.608696  0.611111
1     1  0.478261  0.525000
2     2  0.652174  0.750000
3     3  0.625000  0.688889
4     4  0.666667  0.696296
Mean AUC: 0.6542592592592593

with normative_BLR_BigHC
classification from zscores with standard scaling and residualization on site:
   fold  accuracy   roc_auc
0     0  0.695652  0.730159
1     1  0.608696  0.641667
2     2  0.826087  0.800000
3     3  0.666667  0.592593
4     4  0.583333  0.681481
Mean AUC: 0.6891798941798941

with normative_BLR_BigHC_site_residualized
classification from zscores with standard scaling and residualization on site:
   fold  accuracy   roc_auc
0     0  0.608696  0.690476
1     1  0.565217  0.666667
2     2  0.695652  0.675000
3     3  0.666667  0.570370
4     4  0.750000  0.703704
Mean AUC: 0.6612433862433863


with normative_BLR_OpenBHB
classification from zscores without standard scaling:
fold  accuracy   roc_auc
0     0  0.608696  0.587302
1     1  0.608696  0.516667
2     2  0.521739  0.541667
3     3  0.583333  0.666667
4     4  0.541667  0.607407
Mean AUC: 0.5839417989417989

classification from zscores with standard scaling:
    fold  accuracy   roc_auc
0     0  0.608696  0.626984
1     1  0.521739  0.566667
2     2  0.652174  0.741667
3     3  0.666667  0.651852
4     4  0.666667  0.659259
Mean AUC: 0.6492857142857142

classification with standard scaling and model trained with residualized (on site) roi
    fold  accuracy   roc_auc
0     0  0.565217  0.650794
1     1  0.521739  0.516667
2     2  0.652174  0.733333
3     3  0.666667  0.651852
4     4  0.625000  0.644444
Mean AUC: 0.6394179894179894

"""

"""
df_M0, df_M3 = get_M0_M3()

for roi in SELECTED_REGIONS_OF_INTEREST_RLINK:
    print(roi)
    results = get_roi_diff_percentages(df_M0, df_M3, [roi])
    print(results)

df3 >d0 dans get_roi_diff_percentages

cat12.8
Left Hippocampus_GM_Vol
{0: Left Hippocampus_GM_Vol    86.67
dtype: float64, 1: Left Hippocampus_GM_Vol    93.55
dtype: float64}
Right Hippocampus_GM_Vol
{0: Right Hippocampus_GM_Vol    86.67
dtype: float64, 1: Right Hippocampus_GM_Vol    90.32
dtype: float64}
Left Amygdala_GM_Vol
{0: Left Amygdala_GM_Vol    70.0
dtype: float64, 1: Left Amygdala_GM_Vol    77.42
dtype: float64}
Right Amygdala_GM_Vol
{0: Right Amygdala_GM_Vol    78.33
dtype: float64, 1: Right Amygdala_GM_Vol    83.87
dtype: float64}

(new_env) sp272479@triscotte:/neurospin/signatures/2025_spetiton_rlink_predict_response_anat_backup$ python3 normative_brain_age_models/01_normative_modeling.py 
cat12.7

lHip_GM_Vol
{0: lHip_GM_Vol    24.14
dtype: float64, 1: lHip_GM_Vol    19.35
dtype: float64}
rHip_GM_Vol
{0: rHip_GM_Vol    25.86
dtype: float64, 1: rHip_GM_Vol    19.35
dtype: float64}
lAmy_GM_Vol
{0: lAmy_GM_Vol    20.69
dtype: float64, 1: lAmy_GM_Vol    25.81
dtype: float64}
rAmy_GM_Vol
{0: rAmy_GM_Vol    29.31
dtype: float64, 1: rAmy_GM_Vol    29.03
dtype: float64}


"""
"""
PATHROI_LONG ="/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.8.2_long_roi/neuromorphometrics_cat12_vbm_roi.tsv"
    PATHROI_LONG_280ROI="/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.7_long_roi/neuromorphometrics_cat12_vbm_roi.tsv"
    RESPONSE_DATA_DF_FILE = "/neurospin/rlink/REF_DATABASE/phenotype/ecrf/dataset-outcome_version-4.tsv"

    pop = pd.read_csv(RESPONSE_DATA_DF_FILE, sep='\t')

    dfROI = pd.read_csv(PATHROI_LONG,sep='\t') 
    dfROI=pd.merge(dfROI,pop[["participant_id","Response.Status.at.end.of.follow.up"]],on="participant_id", how="left")
    dfROI=dfROI.rename(columns={'Response.Status.at.end.of.follow.up': "response"})
    dfROI["response"] = dfROI["response"].replace({"GR": 1, "PaR": 0, "NR": 0})
    dfROI = dfROI[dfROI['response'].isin([0, 1])]
    dfROI = dfROI.dropna()


    dfROI280 = pd.read_csv(PATHROI_LONG_280ROI,sep='\t') 
    dfROI280=pd.merge(dfROI280,pop[["participant_id","Response.Status.at.end.of.follow.up"]],on="participant_id", how="left")
    dfROI280=dfROI280.rename(columns={'Response.Status.at.end.of.follow.up': "response"})
    dfROI280["response"] = dfROI280["response"].replace({"GR": 1, "PaR": 0, "NR": 0})
    dfROI280 = dfROI280[dfROI280['response'].isin([0, 1])]
    dfROI280 = dfROI280.dropna()



    print(dfROI)
    print(dfROI280)
    dfROIM3 = dfROI[dfROI["session"]=="M03"]
    dfROIM0 = dfROI[dfROI["session"]=="M00"]
    dfROI280_M3= dfROI280[dfROI280["session"]=="M03"]
    dfROI280_M0=dfROI280[dfROI280["session"]=="M00"]

    results = get_roi_diff_percentages(dfROIM0, dfROIM3, ["Left Hippocampus_GM_Vol"])
    results280= get_roi_diff_percentages(dfROI280_M0, dfROI280_M3, ["lHip_GM_Vol"])
    print(results, "\n", results280)

    results = get_roi_diff_percentages(dfROIM0, dfROIM3, ["Right Hippocampus_GM_Vol"])
    results280= get_roi_diff_percentages(dfROI280_M0, dfROI280_M3, ["rHip_GM_Vol"])
    print(results, "\n", results280)

    results = get_roi_diff_percentages(dfROIM0, dfROIM3, ["Right Amygdala_GM_Vol"])
    results280= get_roi_diff_percentages(dfROI280_M0, dfROI280_M3, ["rAmy_GM_Vol"])
    print(results, "\n", results280)

    results = get_roi_diff_percentages(dfROIM0, dfROIM3, ["Left Amygdala_GM_Vol"])
    results280= get_roi_diff_percentages(dfROI280_M0, dfROI280_M3, ["lAmy_GM_Vol"])
    print(results, "\n", results280)

    df_M0, df_M3 = get_M0_M3()
    for roi in SELECTED_REGIONS_OF_INTEREST_RLINK:
        print(roi)
        results = get_roi_diff_percentages(df_M0, df_M3, [roi])
        print(results)

"""