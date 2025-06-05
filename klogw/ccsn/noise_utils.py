"""
noise_utils.py (LALSuite version, corrected)

Utilities for generating white and colored Gaussian noise for gravitational wave
data analysis using LALSuite. Supports LALSuite PSD models and noise realization.

Author: Your Name
Date: YYYY-MM-DD
"""

import numpy as np
import lal
import lalsimulation as lalsim

def generate_white_noise(N, sigma=1.0, seed=None):
    """
    Generate white Gaussian noise with zero mean and standard deviation sigma.

    Parameters
    ----------
    N : int
        Number of samples.
    sigma : float
        Standard deviation of the white noise.
    seed : int or None
        Random seed for reproducibility. If None, no seeding is done.

    Returns
    -------
    noise : ndarray
        White Gaussian noise series of length N.
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(0, sigma, N)

def generate_colored_noise_from_lalsim(N, fs, psd_model='aLIGOZeroDetHighPower', flow=20.0, seed=None):
    """
    Generate colored noise using a specified LALSuite PSD model from lalsimulation.

    Parameters
    ----------
    N : int
        Number of time-domain samples.
    fs : float
        Sampling rate (Hz).
    psd_model : str
        Name of the LALSuite PSD model. Supported options include:
        'aLIGOZeroDetHighPower', 'aLIGODesignSensitivity', etc.
    flow : float
        Low-frequency cutoff (Hz). Frequencies below flow are zeroed.
    seed : int or None
        Random seed for reproducibility. If None, no seeding is done.

    Returns
    -------
    noise : ndarray
        Time-domain colored Gaussian noise of length N.
    psd_values : ndarray
        One-sided PSD values corresponding to frequency bins (numpy array).
    freq_bins : ndarray
        Frequency array corresponding to PSD values (Hz).
    """
    # Time-domain parameters
    delta_t = 1.0 / fs
    duration = N * delta_t

    # Create an empty REAL8TimeSeries to hold the noise
    # Correct signature: CreateREAL8TimeSeries(name, t0, deltaT, length)
    ts = lal.CreateREAL8TimeSeries(
        "NoiseTS",  # Name
        0,  # GPS start time
        delta_t,  # deltaT (sampling interval)
        N,  # number of points
        lal.DimensionlessUnit,  # sample units
    )

    # Frequency-domain parameters for PSD
    delta_f = 1.0 / duration
    flen = N // 2 + 1    # Number of positive-frequency bins

    # Create a FrequencySeries for the PSD
    if psd_model == 'aLIGOZeroDetHighPower':
        psd_fs = lalsim.SimNoisePSDaLIGOZeroDetHighPower(flen, delta_f, flow, fs/2.0)
    elif psd_model == 'aLIGODesignSensitivity':
        psd_fs = lalsim.SimNoisePSDaLIGOAdvancedLIGO(flen, delta_f, flow, fs/2.0)
    else:
        raise ValueError(f"Unsupported PSD model: {psd_model}")

    # Generate colored Gaussian noise (REAL8) into ts
    noise_ts = lalsim.SimNoiseREAL8TimeSeries(ts, psd_fs, seed if seed is not None else int(np.random.randint(1e9)))

    # Extract numpy array from LAL time series
    noise = np.array([noise_ts.data.data[i] for i in range(noise_ts.data.length)])

    # Extract PSD values into numpy
    psd_values = np.array([psd_fs.data.data[i] for i in range(psd_fs.data.length)])

    # Build frequency bin array
    freq_bins = np.arange(flen) * delta_f

    return noise, psd_values, freq_bins

def plot_noise_psd(freq_bins, psd_values, title="Noise PSD"):
    """
    Plot the one-sided noise PSD obtained from LALSuite.

    Parameters
    ----------
    freq_bins : ndarray
        Frequency array corresponding to PSD values (Hz).
    psd_values : ndarray
        One-sided PSD values (strain^2 / Hz).
    title : str
        Title for the PSD plot.

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
        Plot handles for further customization.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(freq_bins, psd_values, label='One-sided PSD (LALSuite)')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Strain$^2$/Hz')
    ax.set_title(title)
    ax.grid(which='both', linestyle='--', alpha=0.5)
    ax.legend()
    return fig, ax
