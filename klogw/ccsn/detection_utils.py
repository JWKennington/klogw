
"""
detection_utils.py

Utilities for computing the optimal detection statistic (Neyman-Pearson) for 
stochastic CCSN signals, estimating SNR, and evaluating ROC performance.

Author: Your Name
Date: YYYY-MM-DD
"""

import numpy as np
import numpy.linalg as LA

def detection_statistic(x, Cn, Cs):
    """
    Compute the quadratic detection statistic:
    T(x) = x^T [ Cn^{-1} - (Cn + Cs)^{-1} ] x

    Parameters
    ----------
    x : ndarray
        Data vector (time-series strain) of shape (N,).
    Cn : ndarray
        Noise covariance matrix of shape (N, N).
    Cs : ndarray
        Signal covariance matrix of shape (N, N).

    Returns
    -------
    T : float
        Value of the detection statistic.
    """
    # Inverse noise covariance
    Cn_inv = LA.inv(Cn)
    # Inverse total covariance (signal + noise)
    Ctot_inv = LA.inv(Cn + Cs)
    # Compute quadratic form
    T = x.T @ (Cn_inv - Ctot_inv) @ x
    return float(T)

def estimate_snr(Cn, Cs, trials=1000):
    """
    Estimate detection SNR via Monte Carlo:
    SNR = [E(T|H1) - E(T|H0)] / sqrt(Var(T|H0))

    Parameters
    ----------
    Cn : ndarray
        Noise covariance matrix (N x N).
    Cs : ndarray
        Signal covariance matrix (N x N).
    trials : int
        Number of Monte Carlo trials for H0 and H1 each.

    Returns
    -------
    snr_est : float
        Estimated SNR for the detection statistic.
    """
    N = Cn.shape[0]
    Ts_H0 = np.zeros(trials)
    Ts_H1 = np.zeros(trials)
    for i in range(trials):
        # Noise-only sample
        x0 = np.random.multivariate_normal(np.zeros(N), Cn)
        Ts_H0[i] = detection_statistic(x0, Cn, Cs)
        # Signal+noise sample (zero-mean Gaussian with cov Cn + Cs)
        x1 = np.random.multivariate_normal(np.zeros(N), Cn + Cs)
        Ts_H1[i] = detection_statistic(x1, Cn, Cs)
    m0, m1 = Ts_H0.mean(), Ts_H1.mean()
    std0 = Ts_H0.std(ddof=1)
    snr_est = (m1 - m0) / std0
    return snr_est

def compute_roc(Cn, Cs, n_H0=2000, n_H1=2000, thresholds=None):
    """
    Compute ROC curve data for the detection statistic:
    - Generate n_H0 noise-only samples
    - Generate n_H1 signal+noise samples
    - Evaluate detection statistic for each
    - Sweep thresholds to get (FPR, TPR) pairs

    Parameters
    ----------
    Cn : ndarray
        Noise covariance matrix (N x N).
    Cs : ndarray
        Signal covariance matrix (N x N).
    n_H0 : int
        Number of H0 trials (noise only).
    n_H1 : int
        Number of H1 trials (signal + noise).
    thresholds : ndarray or None
        Array of thresholds to evaluate. If None, derive from combined stats.

    Returns
    -------
    fprs : ndarray
        False positive rates at each threshold.
    tprs : ndarray
        True positive rates at each threshold.
    thresholds : ndarray
        Threshold values used.
    """
    N = Cn.shape[0]
    Ts_H0 = np.zeros(n_H0)
    Ts_H1 = np.zeros(n_H1)
    # Sample under H0 and H1
    for i in range(n_H0):
        x0 = np.random.multivariate_normal(np.zeros(N), Cn)
        Ts_H0[i] = detection_statistic(x0, Cn, Cs)
    for i in range(n_H1):
        x1 = np.random.multivariate_normal(np.zeros(N), Cn + Cs)
        Ts_H1[i] = detection_statistic(x1, Cn, Cs)
    # Combine and sort thresholds if not provided
    all_T = np.concatenate([Ts_H0, Ts_H1])
    if thresholds is None:
        thresholds = np.linspace(all_T.min(), all_T.max(), 200)
    fprs = np.zeros_like(thresholds)
    tprs = np.zeros_like(thresholds)
    for idx, thr in enumerate(thresholds):
        fprs[idx] = np.mean(Ts_H0 > thr)
        tprs[idx] = np.mean(Ts_H1 > thr)
    return fprs, tprs, thresholds

def plot_roc(fprs, tprs, label=None, title="ROC Curve"):
    """
    Plot the ROC curve (TPR vs. FPR).

    Parameters
    ----------
    fprs : ndarray
        False positive rates.
    tprs : ndarray
        True positive rates.
    label : str or None
        Label for the ROC curve (legend).
    title : str
        Title for the plot.

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
        Handles for further customization.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fprs, tprs, label=label or "ROC")
    ax.plot([0, 1], [0, 1], 'k--', label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    return fig, ax
