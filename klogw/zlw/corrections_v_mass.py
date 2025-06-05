# driver_vary_template.py

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

# 2) Fix PSD1 and one PSD2 for all templates (e.g. a Gaussian bump at 150 Hz)
PSD1 = load_design_psd(freqs)
PSD2 = make_perturbed_psd(PSD1, freqs, kind="gaussian", center=150.0, width=50.0,
                          amplitude_ratio=0.03)
phi_mp = compute_minimum_phase_phi(PSD2, freqs, N)
W1, W2 = make_whitening_filters(PSD1, PSD2, phi_mp)

# 3) Decide on a list of different (m1,m2) pairs to test
template_list = [
    (20.0, 20.0),
    (30.0, 30.0),
    (40.0, 40.0),
    (30.0, 10.0),
]

n_injections = 200

for (m1, m2) in template_list:
    # 4) Generate FD waveform for this template
    distance_mpc = 500.0
    freqs_fd, H0_fd = generate_fd_waveform(freqs[1], freqs[-1], df, m1, m2, distance_mpc)
    H0 = interpolate_H0(freqs, freqs_fd, H0_fd)

    # 5) Compute the corrections just for this H0
    dt1, dphi1, dt2, dphi2 = compute_corrections(H0, PSD2, phi_mp, freqs)
    print(f"\nTemplate (m1,m2)=({m1},{m2}): dt1={dt1:.3e}  dphi1={dphi1:.3e}")

    # 6) Run injections exactly as before
    t_true_list = []
    t_hat_raw_list = []
    t_hat_corr1_list = []
    t_hat_corr12_list = []

    phi_true_list = []
    phi_hat_raw_list = []
    phi_hat_corr1_list = []
    phi_hat_corr12_list = []

    # Build the (centered) whitened template time series once per block
    H1 = H0 * W1
    h1 = np.fft.ifft(H1, n=N)

    for _ in range(n_injections):
        t_true   = np.random.uniform(0.3, duration - 0.3)
        phi_true = np.random.uniform(0.0, 2.0 * np.pi)

        x2_centered = make_injection_time_series(H0, W2, freqs, N, dt, t_true, phi_true)
        x2_noisy    = x2_centered + 1e-22 * np.random.normal(size=N)

        tc, z_t = match_filter_time_series(h1, x2_noisy, dt)
        idx_peak = np.argmax(np.abs(z_t))
        raw_lag = tc[idx_peak]
        t_hat   = raw_lag % duration
        phi_hat = np.angle(z_t[idx_peak])

        t_hat_corr1  = (t_hat - dt1) % duration
        t_hat_corr12 = (t_hat - (dt1 + dt2)) % duration

        phi_hat_corr1  = (phi_hat - dphi1) % (2.0 * np.pi)
        phi_hat_corr12 = (phi_hat - (dphi1 + dphi2)) % (2.0 * np.pi)

        t_true_list.append(t_true)
        t_hat_raw_list.append(t_hat)
        t_hat_corr1_list.append(t_hat_corr1)
        t_hat_corr12_list.append(t_hat_corr12)

        phi_true_list.append(phi_true)
        phi_hat_raw_list.append(phi_hat)
        phi_hat_corr1_list.append(phi_hat_corr1)
        phi_hat_corr12_list.append(phi_hat_corr12)

    # 7) Compute and plot residuals
    t_true_arr       = np.array(t_true_list)
    t_hat_raw_arr    = np.array(t_hat_raw_list)
    t_hat_corr1_arr  = np.array(t_hat_corr1_list)
    t_hat_corr12_arr = np.array(t_hat_corr12_list)

    phi_true_arr       = np.array(phi_true_list)
    phi_hat_raw_arr    = np.array(phi_hat_raw_list)
    phi_hat_corr1_arr  = np.array(phi_hat_corr1_list)
    phi_hat_corr12_arr = np.array(phi_hat_corr12_list)

    raw_time_res    = wrap_time_residuals(t_hat_raw_arr, t_true_arr, duration)
    corr1_time_res  = wrap_time_residuals(t_hat_corr1_arr, t_true_arr, duration)
    corr12_time_res = wrap_time_residuals(t_hat_corr12_arr, t_true_arr, duration)

    raw_phase_res    = wrap_phase_residuals(phi_hat_raw_arr, phi_true_arr)
    corr1_phase_res  = wrap_phase_residuals(phi_hat_corr1_arr, phi_true_arr)
    corr12_phase_res = wrap_phase_residuals(phi_hat_corr12_arr, phi_true_arr)

    print(f"Plotting for template (m1,m2)=({m1},{m2}) …")
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
