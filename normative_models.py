"""
Normative model: one Bayesian Linear Regression (BLR) per brain ROI.

For each ROI, the model learns the expected value (mean) and spread (std of residuals)
of the ROI volume in healthy controls, as a function of age (B-spline basis) and sex.
A new subject's z-score for an ROI is then:

    z = (observed volume - predicted healthy mean) / residual std

Optionally, each ROI is first passed through a sinh-arcsinh warping (fitted by maximum
likelihood) so that non-Gaussian ROI distributions can be modelled.

"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


# ---------------------------------------------------------------------------
# sinh-arcsinh warping
# ---------------------------------------------------------------------------
def warp_sinh_arcsinh(y, epsilon=0.0, b=1.0):
    return np.sinh(b * np.arcsinh(y) + epsilon * b)


def inv_warp_sinh_arcsinh(gy, epsilon=0.0, b=1.0):
    return np.sinh((np.arcsinh(gy) - epsilon * b) / b)


def neg_log_likelihood_sinh_arcsinh(params, y):
    """Negative log-likelihood of y under a standard normal after warping (incl. Jacobian)."""
    epsilon, b = params
    if b <= 0:
        return np.inf
    g_y = warp_sinh_arcsinh(y, epsilon, b)
    dg_dy = b * np.cosh(b * np.arcsinh(y) + epsilon * b) / np.sqrt(1 + y**2)
    return -np.sum(norm.logpdf(g_y) + np.log(dg_dy))


# ---------------------------------------------------------------------------
# Normative model
# ---------------------------------------------------------------------------
class NormativeBLR(BaseEstimator, RegressorMixin):
    """
    Parameters
    ----------
    warp : bool
        Apply a sinh-arcsinh warping to each ROI before fitting.
    bsplines : bool
        Model age with a cubic B-spline basis (and sex as a categorical variable).
        If False, age and sex are used as-is (sex must then be numeric).
    n_knots : int
        Number of knots of the age spline.
    """

    def __init__(self, warp=False, bsplines=True, n_knots=3):
        self.warp = warp
        self.bsplines = bsplines
        self.n_knots = n_knots

    def _make_pipeline(self):
        if self.bsplines:
            preprocessor = ColumnTransformer([
                ("spline_age", SplineTransformer(degree=3, n_knots=self.n_knots), [0]),
                ("onehot_sex", OneHotEncoder(drop="if_binary"), [1]), # treat sex as categorical variable
            ])
        else:
            preprocessor = "passthrough"
        return make_pipeline(preprocessor, StandardScaler(), BayesianRidge())

    def fit(self, X_covariates, Y_roi):
        """
        Fit one model per ROI.

        X_covariates : DataFrame (n_subjects, 2), columns [age, sex], sex in {"female", "male"}
        Y_roi        : array (n_subjects, n_rois)
        """
        self.models_, self.stds_, self.epsilons_, self.betas_ = [], [], [], []

        for i in range(Y_roi.shape[1]):
            y = Y_roi[:, i]

            # optional warping
            epsilon, beta = 0.0, 1.0
            if self.warp:
                res = minimize(neg_log_likelihood_sinh_arcsinh, [0.0, 1.0], args=(y,),
                               method="L-BFGS-B", bounds=[(-2, 2), (1e-3, 10)])
                epsilon, beta = res.x
                y_fit = warp_sinh_arcsinh(y, epsilon, beta)
            else:
                y_fit = y

            model = self._make_pipeline()
            model.fit(X_covariates, y_fit)

            # residual std, computed in the original (unwarped) space
            y_pred = model.predict(X_covariates)
            if self.warp:
                y_pred = inv_warp_sinh_arcsinh(y_pred, epsilon, beta)
            resid_std = np.std(y - y_pred, ddof=1)

            self.models_.append(model)
            self.stds_.append(resid_std)
            self.epsilons_.append(epsilon)
            self.betas_.append(beta)

        return self

    def predict(self, X_covariates, Y_roi=None, return_zscores=False):
        """
        Predict the healthy mean of each ROI, and optionally the z-scores of Y_roi.

        X_covariates : DataFrame (n_subjects, 2), columns [age, sex]
        Y_roi        : array (n_subjects, n_rois), required if return_zscores=True,
                       with ROIs in the same order as during fit.

        Returns preds (n_subjects, n_rois) [, zscores (n_subjects, n_rois)]
        """
        if return_zscores and Y_roi is None:
            raise ValueError("Y_roi must be provided to compute z-scores")

        preds, zscores = [], []
        for i, model in enumerate(self.models_):
            y_pred = model.predict(X_covariates)
            if self.warp:
                y_pred = inv_warp_sinh_arcsinh(y_pred, self.epsilons_[i], self.betas_[i])
            preds.append(y_pred)

            if return_zscores:
                zscores.append((Y_roi[:, i] - y_pred) / self.stds_[i])

        preds = np.vstack(preds).T
        if return_zscores:
            return preds, np.vstack(zscores).T
        return preds
