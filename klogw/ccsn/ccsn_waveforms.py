
"""
ccsn_waveforms.py

Module for generating stochastic CCSN-like gravitational wave signals with a deterministic
time-varying frequency trajectory omega(t). Provides functions to define frequency models,
generate noise-modulated waveforms, and visualize spectrograms.

Author: Your Name
Date: YYYY-MM-DD
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def omega_sqrt_model(t, f0, alpha):
    """
    Example frequency trajectory: f(t) = f0 + alpha * sqrt(t).

    Parameters
    ----------
    t : ndarray
        Time array (seconds).
    f0 : float
        Base frequency at t = 0 (Hz).
    alpha : float
        Coefficient for sqrt(t) (Hz / sqrt(s)).

    Returns
    -------
    f_t : ndarray
        Instantaneous frequency at each time t (Hz).
    """
    return f0 + alpha * np.sqrt(t)

def omega_exponential_model(t, f0, delta, beta):
    """
    Example frequency trajectory: f(t) = f0 + delta * (1 - exp(-beta * t)).

    Parameters
    ----------
    t : ndarray
        Time array (seconds).
    f0 : float
        Base frequency at t = 0 (Hz).
    delta : float
        Asymptotic frequency increase (Hz).
    beta : float
        Decay rate for exponential (1/s).

    Returns
    -------
    f_t : ndarray
        Instantaneous frequency at each time t (Hz).
    """
    return f0 + delta * (1 - np.exp(-beta * t))

def omega_power_model(t, f0, gamma, p):
    """
    Example frequency trajectory: f(t) = f0 + gamma * t^p, 0 < p < 1.

    Parameters
    ----------
    t : ndarray
        Time array (seconds).
    f0 : float
        Base frequency at t = 0 (Hz).
    gamma : float
        Coefficient for t^p (Hz / s^p).
    p : float
        Power exponent (0 < p < 1).

    Returns
    -------
    f_t : ndarray
        Instantaneous frequency at each time t (Hz).
    """
    return f0 + gamma * t**p

def omega_logarithmic_model(t, f0, kappa, nu):
    """
    Example frequency trajectory: f(t) = f0 + kappa * log(1 + nu * t).

    Parameters
    ----------
    t : ndarray
        Time array (seconds).
    f0 : float
        Base frequency at t = 0 (Hz).
    kappa : float
        Scaling coefficient for log term (Hz).
    nu : float
        Rate parameter inside log (1/s).

    Returns
    -------
    f_t : ndarray
        Instantaneous frequency at each time t (Hz).
    """
    return f0 + kappa * np.log1p(nu * t)

def generate_ccsn_waveform(fs, T, omega_func, omega_params, seed=None):
    """
    Generate a stochastic CCSN-like waveform by modulating white noise with a time-varying
    frequency trajectory omega(t). The output is the real part of the analytic noise signal
    multiplied by exp(i * phi(t)), where phi(t) = 2*pi * integral(f(t) dt).

    Parameters
    ----------
    fs : float
        Sampling rate (Hz).
    T : float
        Duration of the waveform (seconds).
    omega_func : callable
        Function taking (t, *omega_params) and returning instantaneous frequency f(t) (Hz).
    omega_params : tuple
        Parameters for the omega_func.
    seed : int or None
        Random seed for reproducibility. If None, no seeding is done.

    Returns
    -------
    t : ndarray
        Time array of shape (N,) where N = int(T * fs).
    signal : ndarray
        Generated time-domain strain series of shape (N,).
    f_t : ndarray
        Instantaneous frequency array of shape (N,) (Hz).
    """
    if seed is not None:
        np.random.seed(seed)

    N = int(T * fs)
    t = np.linspace(0, T, N, endpoint=False)

    # Compute instantaneous frequency f(t) using the provided model
    f_t = omega_func(t, *omega_params)

    # Integrate f(t) to get phase phi(t) = 2*pi * ∫ f(t) dt
    # We approximate integral with cumulative sum
    phi_t = 2 * np.pi * np.cumsum(f_t) / fs

    # Generate white Gaussian noise (real)
    white = np.random.normal(0, 1, N)

    # Compute analytic signal (Hilbert transform) to get complex noise
    analytic_noise = sig.hilbert(white)

    # Frequency-modulate the noise: multiply analytic signal by exp(i * phi(t))
    modulated = analytic_noise * np.exp(1j * phi_t)

    # Take the real part as the final strain series
    signal = np.real(modulated)

    return t, signal, f_t

def plot_spectrogram(signal, fs, title="Spectrogram", cmap="viridis", fmax=None):
    """
    Plot the spectrogram of a time-domain signal.

    Parameters
    ----------
    signal : ndarray
        Time-domain strain series.
    fs : float
        Sampling rate (Hz).
    title : str
        Title for the spectrogram plot.
    cmap : str
        Colormap for the spectrogram (e.g. 'viridis', 'plasma').
    fmax : float or None
        Maximum frequency to display (Hz). If None, use fs/2.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the spectrogram.
    ax : matplotlib.axes.Axes
        Axes object containing the plot.
    """
    # Compute Short-Time Fourier Transform (STFT)
    nperseg = 256  # window length for STFT (samples)
    noverlap = nperseg // 2
    f, t_seg, Sxx = sig.spectrogram(signal, fs, window='hann',
                                    nperseg=nperseg, noverlap=noverlap,
                                    scaling='density', mode='magnitude')

    # Convert to decibels
    Sxx_db = 20 * np.log10(Sxx + 1e-12)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    if fmax is None:
        fmax = fs / 2
    im = ax.pcolormesh(t_seg, f, Sxx_db, shading='auto', cmap=cmap)
    ax.set_ylim(0, fmax)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Amplitude [dB]')
    return fig, ax
