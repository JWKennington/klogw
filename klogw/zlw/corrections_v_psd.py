"""Refactored version of v6"""

# driver_vary_psd.py

import numpy as np
from klogw.zlw.utils import (
    load_design_psd,
    make_perturbed_psd,
    compute_minimum_phase_phi,
    make_whitening_filters,
    generate_fd_waveform,
    interpolate_H0,
    compute_corrections,
    match_filter_time_series,
    make_injection_time_series,
    plot_residuals_scatter,
    wrap_time_residuals,
    wrap_phase_residuals,
    plot_residuals_histograms,
)

# -------------------- Set common parameters --------------------
fs = 16384.0              # [Hz]
duration = 20.0           # [s]
N = int(fs * duration)
dt = 1.0 / fs
df = 1.0 / duration

# Build the one-sided freq axis
freqs = np.fft.rfftfreq(N, d=dt)

# 1) Base PSD1 (design)
PSD1 = load_design_psd(freqs)

# 2) Generate a single “base” template waveform for (m1=30,m2=30)
m1, m2      = 30.0, 30.0   # solar masses
distance_mpc = 500.0       # injection distance (affects SNR)
f_min        = freqs[1]
f_max        = freqs[-1]

freqs_fd, H0_fd = generate_fd_waveform(f_min, f_max, df, m1, m2, distance_mpc)
H0             = interpolate_H0(freqs, freqs_fd, H0_fd)

# Now loop over different PSD perturbations
perturbation_configs = [
    {"kind": "gaussian", "center": 50.0,  "width":  20.0, "amplitude_ratio": 0.3},
    {"kind": "gaussian", "center":100.0,  "width":  50.0, "amplitude_ratio": 0.3},
    {"kind": "gaussian", "center":200.0,  "width": 100.0, "amplitude_ratio": 0.3},
    {"kind": "random",   "amplitude_range": 0.3},
]

n_injections = 200

for cfg in perturbation_configs:
    # 3) Build PSD2 from PSD1 with this perturbation
    PSD2 = make_perturbed_psd(PSD1, freqs, **cfg)

    # 4) Compute phi_mp from PSD2
    phi_mp = compute_minimum_phase_phi(PSD2, freqs, N)

    # 5) Whitening filters
    W1, W2 = make_whitening_filters(PSD1, PSD2, phi_mp)

    # 6) Compute Δt1, Δφ1, Δt2, Δφ2 once
    dt1, dphi1, dt2, dphi2 = compute_corrections(H0, PSD2, phi_mp, freqs)
    print("\nPSD config:", cfg)
    print(f"   Delta t1 = {dt1:.3e} s,  Delta phi1 = {dphi1:.3e} rad")
    print(f"   Delta t2 = {dt2:.3e} s,  Delta phi2 = {dphi2:.3e} rad")

    # Storage for residuals
    t_true_list = []
    t_hat_raw_list = []
    t_hat_corr1_list = []
    t_hat_corr12_list = []

    phi_true_list = []
    phi_hat_raw_list = []
    phi_hat_corr1_list = []
    phi_hat_corr12_list = []

    # 7) Do N injections
    for _ in range(n_injections):
        # (a) pick random true time/phase
        t_true   = np.random.uniform(0.3, duration - 0.3)
        phi_true = np.random.uniform(0.0, 2.0 * np.pi)

        # (b) Build whitened time-series injection x2(t), add small white noise
        x2_centered = make_injection_time_series(H0, W2, freqs, N, dt, t_true, phi_true)
        # Add a tiny whitened Gaussian noise so the MF peak “wiggles”
        x2_noisy = x2_centered + 1e-22 * np.random.normal(size=N)

        # (c) Build a pre‐centered “template” h1(t) once per PSD config
        if _ == 0:
            # h1_base: FD whitened template (t_true=0, φ_true=0)
            H1_base = H0 * W1
            h1_base = np.fft.ifft(H1_base, n=N)

        # (d) Match‐filter
        tc, z_t_complex = match_filter_time_series(h1_base, x2_noisy, dt)
        idx_peak = np.argmax(np.abs(z_t_complex))
        raw_lag  = tc[idx_peak]
        t_hat    = raw_lag % duration
        phi_hat  = np.angle(z_t_complex[idx_peak])

        # (e) Apply analytic corrections
        t_hat_corr1  = (t_hat - dt1) % duration
        t_hat_corr12 = (t_hat - (dt1 + dt2)) % duration

        phi_hat_corr1  = (phi_hat - dphi1) % (2.0*np.pi)
        phi_hat_corr12 = (phi_hat - (dphi1 + dphi2)) % (2.0*np.pi)

        # (f) Store
        t_true_list.append(t_true)
        t_hat_raw_list.append(t_hat)
        t_hat_corr1_list.append(t_hat_corr1)
        t_hat_corr12_list.append(t_hat_corr12)

        phi_true_list.append(phi_true)
        phi_hat_raw_list.append(phi_hat)
        phi_hat_corr1_list.append(phi_hat_corr1)
        phi_hat_corr12_list.append(phi_hat_corr12)

    # 8) Convert to arrays, compute residuals, and histogram‐plot
    t_true_arr       = np.array(t_true_list)
    t_hat_raw_arr    = np.array(t_hat_raw_list)
    t_hat_corr1_arr  = np.array(t_hat_corr1_list)
    t_hat_corr12_arr = np.array(t_hat_corr12_list)

    phi_true_arr       = np.array(phi_true_list)
    phi_hat_raw_arr    = np.array(phi_hat_raw_list)
    phi_hat_corr1_arr  = np.array(phi_hat_corr1_list)
    phi_hat_corr12_arr = np.array(phi_hat_corr12_list)

    raw_time_res    = wrap_time_residuals(t_hat_raw_arr,    t_true_arr, duration)
    corr1_time_res  = wrap_time_residuals(t_hat_corr1_arr,  t_true_arr, duration)
    corr12_time_res = wrap_time_residuals(t_hat_corr12_arr, t_true_arr, duration)

    raw_phase_res    = wrap_phase_residuals(phi_hat_raw_arr,    phi_true_arr)
    corr1_phase_res  = wrap_phase_residuals(phi_hat_corr1_arr,  phi_true_arr)
    corr12_phase_res = wrap_phase_residuals(phi_hat_corr12_arr, phi_true_arr)

    # 9) Finally plot
    plot_residuals_histograms(
        raw_time_res, corr1_time_res, corr12_time_res,
        raw_phase_res, corr1_phase_res, corr12_phase_res
    )

    # 10) SCATTERPLOTS of true vs. estimated
    plot_residuals_scatter(
        t_true_arr,
        t_hat_raw_arr,
        t_hat_corr1_arr,
        t_hat_corr12_arr,
        phi_true_arr,
        phi_hat_raw_arr,
        phi_hat_corr1_arr,
        phi_hat_corr12_arr,
        duration,
    )
