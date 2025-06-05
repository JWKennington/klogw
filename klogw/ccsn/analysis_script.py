"""
analysis_script.py

Example script demonstrating:
1. Generation of CCSN-like waveforms using deterministic omega(t) models.
2. Addition of white or colored noise using LALSuite-based utilities.
3. Computation of spectrograms to validate waveform.
4. Calculation of the Neyman-Pearson detection statistic, SNR estimation, and ROC curves.
5. Sensitivity analysis to perturbations in omega(t).

Usage:
    python analysis_script.py

Dependencies:
    numpy, scipy, matplotlib, lalsuite

Author: Your Name
Date: YYYY-MM-DD
"""

import numpy as np
import matplotlib.pyplot as plt

# Import module functions
from ccsn_waveforms import (
    omega_sqrt_model, omega_exponential_model, omega_power_model, omega_logarithmic_model,
    generate_ccsn_waveform, plot_spectrogram
)
from noise_utils import generate_white_noise, generate_colored_noise_from_lalsim, plot_noise_psd
from detection_utils import detection_statistic, estimate_snr, compute_roc, plot_roc

def main():
    # === 1. Generate CCSN-like waveform ===
    fs = 4096             # sampling rate [Hz]
    T = 1.0               # duration [s]
    f0, alpha = 100.0, 500.0  # parameters for sqrt frequency model

    # Generate baseline waveform
    t, signal_clean, f_t = generate_ccsn_waveform(
        fs, T, omega_sqrt_model, (f0, alpha), seed=42
    )

    # Plot the clean waveform and its instantaneous frequency
    plt.figure(figsize=(8, 3))
    plt.plot(t, signal_clean, label="Stochastic CCSN Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Strain (arb. units)")
    plt.title("Generated CCSN-like Waveform (Clean)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Validate via spectrogram
    fig, ax = plot_spectrogram(signal_clean, fs, title="Signal Spectrogram")
    # Overlay the instantaneous frequency trajectory
    ax.plot(t, f_t, 'w--', label=f"f(t) = {f0} + {alpha}√t")
    ax.legend(loc='upper left', fontsize='small')
    plt.show()

    # === 2. Generate Noise (White and Colored) ===
    # White Gaussian noise example
    N = len(signal_clean)
    white_noise = generate_white_noise(N, sigma=1.0, seed=1)

    # Colored noise using LALSuite PSD
    colored_noise, psd_values, freq_bins = generate_colored_noise_from_lalsim(
        N, fs, psd_model='aLIGOZeroDetHighPower', flow=20.0, seed=1
    )

    # Plot the PSD
    fig, ax = plot_noise_psd(freq_bins, psd_values, title="Advanced LIGO Design PSD (LALSuite)")
    plt.show()

    # === 3. Create Data Streams: Signal + Noise ===
    # Scale signal to achieve desired amplitude (e.g. peak ~0.5 of noise RMS)
    scale_factor = 0.5 / np.std(signal_clean)
    signal_scaled = scale_factor * signal_clean

    # Data in white noise
    data_white = signal_scaled + white_noise

    # Data in colored noise
    data_colored = signal_scaled + colored_noise

    # Plot a short segment of data in colored noise
    plt.figure(figsize=(8, 3))
    plt.plot(t[:1024], data_colored[:1024], color='gray', label="Colored noise + Signal")
    plt.plot(t[:1024], signal_scaled[:1024], color='r', alpha=0.7, label="Scaled Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Strain")
    plt.title("Signal in Colored Noise (First 1024 Samples)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === 4. Build Covariance Matrices ===
    # For simplicity: Cn_white = sigma^2 * I
    sigma_white = 1.0  # white noise standard deviation
    Cn_white = np.eye(N) * sigma_white**2

    # Estimate Cs (signal covariance) via outer product of waveform
    Cs_est = np.outer(signal_scaled, signal_scaled)

    # === 5. Compute Detection Statistic ===
    # For white noise case
    T_white = detection_statistic(data_white, Cn_white, Cs_est)
    print(f"Detection statistic T (white noise) = {T_white:.3f}")

    # Monte Carlo SNR estimate
    snr_white = estimate_snr(Cn_white, Cs_est, trials=500)
    print(f"Estimated SNR (white noise) = {snr_white:.3f}")

    # === 6. ROC Curve for White Noise Detector ===
    fprs, tprs, thresh = compute_roc(Cn_white, Cs_est, n_H0=2000, n_H1=2000)
    fig, ax = plot_roc(fprs, tprs, label="White noise ROC")
    plt.show()

    # === 7. Sensitivity to Omega(t) Perturbations ===
    # Baseline model: f0=100, alpha=500
    # Perturbed model: 20% higher alpha
    alt_alpha = alpha * 1.2
    _, signal_alt, _ = generate_ccsn_waveform(fs, T, omega_sqrt_model, (f0, alt_alpha), seed=43)
    signal_alt_scaled = scale_factor * signal_alt

    # Data for perturbed model in white noise
    data_alt_white = signal_alt_scaled + generate_white_noise(N, sigma=sigma_white, seed=2)
    T_alt = detection_statistic(data_alt_white, Cn_white, Cs_est)
    print(f"Detection statistic T (perturbed α=1.2α₀) = {T_alt:.3f}")

    # Estimate SNR for perturbed model vs. baseline detector
    snr_alt = estimate_snr(Cn_white, Cs_est, trials=500)
    print(f"Estimated SNR (perturbed model, white noise) = {snr_alt:.3f}")

if __name__ == "__main__":
    main()
