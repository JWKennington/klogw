# driver_vary_snr.py

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

# 1) Common setup
fs = 16384.0
duration = 20.0
N = int(fs * duration)
dt = 1.0 / fs
df = 1.0 / duration
freqs = np.fft.rfftfreq(N, d=dt)

# 2) Fix PSD1, PSD2 (one bump at 150 Hz)
PSD1 = load_design_psd(freqs)
PSD2 = make_perturbed_psd(PSD1, freqs, kind="gaussian", center=150.0, width=50.0, amplitude_ratio=0.3)
phi_mp = compute_minimum_phase_phi(PSD2, freqs, N)
W1, W2 = make_whitening_filters(PSD1, PSD2, phi_mp)

# 3) Pick a single template mass pair (m1=30,m2=30) and generate H0 once
m1, m2 = 30.0, 30.0
freqs_fd, H0_fd = generate_fd_waveform(freqs[1], freqs[-1], df, m1, m2, distance_mpc=500.0)
H0 = interpolate_H0(freqs, freqs_fd, H0_fd)
dt1, dphi1, dt2, dphi2 = compute_corrections(H0, PSD2, phi_mp, freqs)

# 4) Define a range of distances (=> different SNR)
distances_mpc = [200.0, 500.0, 1000.0, 2000.0]  # farther => lower SNR

for dist in distances_mpc:
    print(f"\n--- Distance = {dist} Mpc ---")
    # Regenerate H0 with new distance so amplitude (and SNR) changes:
    freq_fd2, H0_fd2 = generate_fd_waveform(freqs[1], freqs[-1], df, m1, m2, dist)
    H0_scaled = interpolate_H0(freqs, freq_fd2, H0_fd2)

    # Compute corrections for this scaled H0:
    dt1_d, dphi1_d, dt2_d, dphi2_d = compute_corrections(H0_scaled, PSD2, phi_mp, freqs)
    print(f"   Δt₁(d)={dt1_d:.3e}  Δφ₁(d)={dphi1_d:.3e}")

    # Storage for residuals
    t_true_list = []
    t_hat_raw_list = []
    t_hat_corr1_list = []
    t_hat_corr12_list = []

    phi_true_list = []
    phi_hat_raw_list = []
    phi_hat_corr1_list = []
    phi_hat_corr12_list = []

    # Build centered template once
    H1 = H0_scaled * W1
    h1 = np.fft.ifft(H1, n=N)

    for _ in range(200):
        t_true   = np.random.uniform(0.3, duration - 0.3)
        phi_true = np.random.uniform(0.0, 2.0 * np.pi)

        x2_centered = make_injection_time_series(H0_scaled, W2, freqs, N, dt, t_true, phi_true)
        # Add noise at a level scaled by distance (farther -> smaller amplitude -> effectively lower SNR)
        noise_sigma = 1e-22 * (dist / 500.0)  # scale noise so SNR changes
        x2_noisy    = x2_centered + noise_sigma * np.random.normal(size=N)

        tc, z_t = match_filter_time_series(h1, x2_noisy, dt)
        idx_peak = np.argmax(np.abs(z_t))
        raw_lag = tc[idx_peak]
        t_hat   = raw_lag % duration
        phi_hat = np.angle(z_t[idx_peak])

        t_hat_corr1  = (t_hat - dt1_d) % duration
        t_hat_corr12 = (t_hat - (dt1_d + dt2_d)) % duration

        phi_hat_corr1  = (phi_hat - dphi1_d) % (2.0 * np.pi)
        phi_hat_corr12 = (phi_hat - (dphi1_d + dphi2_d)) % (2.0 * np.pi)

        t_true_list.append(t_true)
        t_hat_raw_list.append(t_hat)
        t_hat_corr1_list.append(t_hat_corr1)
        t_hat_corr12_list.append(t_hat_corr12)

        phi_true_list.append(phi_true)
        phi_hat_raw_list.append(phi_hat)
        phi_hat_corr1_list.append(phi_hat_corr1)
        phi_hat_corr12_list.append(phi_hat_corr12)

    # Compute and plot residuals
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
    corr1_phase_res  = wrap_phase_residuals(phi_hat_corr1_arr,    phi_true_arr)
    corr12_phase_res = wrap_phase_residuals(phi_hat_corr12_arr, phi_true_arr)

    print(f"Plotting for distance = {dist} Mpc …")
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
