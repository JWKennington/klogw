#!/usr/bin/env python
"""
driver_more_realistic.py

A “realistic” injection‐and‐recovery driver that demonstrates how timing and phase
residuals behave when we allow:
  (1) PSD errors,
  (2) template‐intrinsic variation,
  (3) injection‐intrinsic ≠ template‐intrinsic,
  (4) PSD‐shaped noise.

You can toggle each of the four categories on or off by setting the boolean flags below.

Dependencies:
  • numpy
  • matplotlib
  • lalsuite  (lal, lalsimulation)
  • zlw_utils.py (the helper functions we wrote above)
"""

import numpy as np
import matplotlib.pyplot as plt

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
    wrap_time_residuals,
    wrap_phase_residuals,
    plot_residuals_histograms,
    plot_residuals_scatter,
)

# ──────────────────────────────────────────────────────────────────────────────
# (A) USER‐CONFIGURABLE FLAGS
# ──────────────────────────────────────────────────────────────────────────────

VARY_PSD                 = True
VARY_TEMPLATE_INTRINSICS = True
VARY_INJ_INTRINSICS      = False  # now OFF, so injection masses = template masses
ADD_PSD_SHAPED_NOISE     = True

# ──────────────────────────────────────────────────────────────────────────────
# (B) FIXED RUN PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

fs       = 16384.0
duration = 20.0
N        = int(fs * duration)
dt       = 1.0 / fs
df       = 1.0 / duration

freqs = np.fft.rfftfreq(N, d=dt)  # length M = N//2 + 1

TEMPLATE_M1_RANGE    = (10.0, 100.0)
TEMPLATE_M2_RANGE    = (10.0, 100.0)
INJ_MASS_OFFSET_STD  = 1.0       # ±1 M⊙ Gaussian scattering

n_injections = 500

PSD2_FIXED_KIND   = "gaussian"
PSD2_FIXED_KWARGS = {"center":150.0, "width":50.0, "amplitude_ratio":0.3}

PERTURBATION_LIST = [
    {"kind":"gaussian","center":  50.0, "width":  20.0, "amplitude_ratio":0.3},
    {"kind":"gaussian","center": 100.0, "width":  50.0, "amplitude_ratio":0.3},
    {"kind":"gaussian","center": 200.0, "width": 100.0, "amplitude_ratio":0.3},
    {"kind":"random",  "amplitude_range":0.3},
]

BASE_M1 = 30.0
BASE_M2 = 30.0
BASE_DISTANCE_MPC = 100.0


# ──────────────────────────────────────────────────────────────────────────────
# (C) HELPER TO GENERATE PSD‐SHAPED WHITENED NOISE
# ──────────────────────────────────────────────────────────────────────────────

def make_whitened_noise(PSD2, dt):
    """
    Return one realization of a *whitened*, zero‐mean Gaussian noise time series
    of length N_full = 2*(M−1), given the one‐sided PSD2[freqs]. The output has
    the property that, in the freq domain, sqrt(PSD2)*FFT(noise) is white.
    """
    M      = PSD2.shape[0]         # = N//2 + 1
    N_full = (M - 1) * 2
    df_full = 1.0 / (N_full * dt)

    re = np.random.normal(size=M)
    im = np.random.normal(size=M)
    noise_fd_pos = (re + 1j * im) / np.sqrt(2.0 * df_full)

    noise_fd_pos *= 1.0 / np.sqrt(PSD2)

    noise_fd_full = np.zeros(N_full, dtype=complex)
    noise_fd_full[:M] = noise_fd_pos
    noise_fd_full[M:] = np.conj(noise_fd_pos[-2:0:-1])

    noise_t = np.fft.ifft(noise_fd_full)
    return noise_t.real


# ──────────────────────────────────────────────────────────────────────────────
# (D) LOAD DESIGN PSD1 ON freqs
# ──────────────────────────────────────────────────────────────────────────────

PSD1 = load_design_psd(freqs)


# ──────────────────────────────────────────────────────────────────────────────
# (E) PREALLOCATE STORAGE FOR RESIDUALS
# ──────────────────────────────────────────────────────────────────────────────

t_true_list       = []
t_hat_raw_list    = []
t_hat_corr1_list  = []
t_hat_corr12_list = []

phi_true_list       = []
phi_hat_raw_list    = []
phi_hat_corr1_list  = []
phi_hat_corr12_list = []


# ──────────────────────────────────────────────────────────────────────────────
# (F) MAIN INJECTION LOOP
# ──────────────────────────────────────────────────────────────────────────────

for i in range(n_injections):
    # ─────────────────────────────────────────────────────────────────────────
    # (F.1) RANDOM COALESCENCE TIME & PHASE
    # ─────────────────────────────────────────────────────────────────────────
    t_true   = np.random.uniform(0.3, duration - 0.3)
    phi_true = np.random.uniform(0.0, 2.0 * np.pi)

    # ─────────────────────────────────────────────────────────────────────────
    # (F.2) POSSIBLY VARY PSD2
    # ─────────────────────────────────────────────────────────────────────────
    if VARY_PSD:
        cfg  = PERTURBATION_LIST[np.random.randint(len(PERTURBATION_LIST))]
        PSD2 = make_perturbed_psd(PSD1, freqs, **cfg)
    else:
        PSD2 = make_perturbed_psd(PSD1, freqs, **{**{"kind":PSD2_FIXED_KIND}, **PSD2_FIXED_KWARGS})

    # Compute minimum‐phase φ_mp from PSD2 and build whitening filters
    phi_mp = compute_minimum_phase_phi(PSD2, freqs, N)
    W1, W2 = make_whitening_filters(PSD1, PSD2, phi_mp)

    # ─────────────────────────────────────────────────────────────────────────
    # (F.3) CHOOSE TEMPLATE MASSES (m1_template, m2_template)
    # ─────────────────────────────────────────────────────────────────────────
    if VARY_TEMPLATE_INTRINSICS:
        m1_template = np.random.uniform(*TEMPLATE_M1_RANGE)
        m2_template = np.random.uniform(*TEMPLATE_M2_RANGE)
    else:
        m1_template = BASE_M1
        m2_template = BASE_M2

    freqs_fd_temp, H0_fd_temp = generate_fd_waveform(
        freqs[1], freqs[-1], df, m1_template, m2_template, BASE_DISTANCE_MPC
    )
    H0_template = interpolate_H0(freqs, freqs_fd_temp, H0_fd_temp)
    # (interpolate_H0 preserves phase correctly)

    # ─────────────────────────────────────────────────────────────────────────
    # (F.4) CHOOSE INJECTION MASSES (m1_inj, m2_inj)
    # ─────────────────────────────────────────────────────────────────────────
    if VARY_INJ_INTRINSICS:
        m1_inj = m1_template + np.random.normal(scale=INJ_MASS_OFFSET_STD)
        m2_inj = m2_template + np.random.normal(scale=INJ_MASS_OFFSET_STD)
    else:
        m1_inj = m1_template
        m2_inj = m2_template

    m1_inj = max(1.0, m1_inj)
    m2_inj = max(1.0, m2_inj)

    freqs_fd_inj, H0_fd_inj = generate_fd_waveform(
        freqs[1], freqs[-1], df, m1_inj, m2_inj, BASE_DISTANCE_MPC
    )
    H0_inj = interpolate_H0(freqs, freqs_fd_inj, H0_fd_inj)

    # ─────────────────────────────────────────────────────────────────────────
    # (F.5) COMPUTE ANALYTIC CORRECTIONS FROM THE TEMPLATE
    # ─────────────────────────────────────────────────────────────────────────
    dt1, dphi1, dt2, dphi2 = compute_corrections(H0_template, PSD2, phi_mp, freqs)

    # ─────────────────────────────────────────────────────────────────────────
    # (F.6) BUILD FULLY COMPLEX “WHITENED TEMPLATE” h1(t)
    # ─────────────────────────────────────────────────────────────────────────
    #  → Instead of mirroring manually, call ifft on the one‐sided H1_pos.
    #  → That returns a *complex* analytic template.  (Just as in v5.)
    #
    #  H1_pos has length M = N//2 + 1:
    H1_pos = H0_template * W1         # length = M
    h1     = np.fft.ifft(H1_pos, n=N) # length = N, complex‐valued

    # ─────────────────────────────────────────────────────────────────────────
    # (F.7) BUILD “WHITENED INJECTION” x2(t) via make_injection_time_series
    # ─────────────────────────────────────────────────────────────────────────
    x2_clean = make_injection_time_series(H0_inj, W2, freqs, N, dt, t_true, phi_true)

    # Add PSD‐shaped noise so that post‐noise SNR ~ 10
    if ADD_PSD_SHAPED_NOISE:
        noise_t_raw = make_whitened_noise(PSD2, dt)  # time‐domain RMS is “large”
        noise_rms   = np.std(noise_t_raw)
        if noise_rms == 0.0:
            noise_rms = 1.0
        # Normalize to RMS=1, then scale down to 0.1
        noise_t_unit = (noise_t_raw / noise_rms) * 0.1

        # (F.7.2) measure zero‐noise matched‐filter SNR of x2_clean
        tc0, z0 = match_filter_time_series(h1, x2_clean, dt)
        snr0 = np.max(np.abs(z0))  # could be tiny if waveform is weak

        # (F.7.3) boost signal so snr0 → SNR_target if needed
        SNR_target = 10.0
        if snr0 > 0:
            scale_signal = SNR_target / snr0
        else:
            scale_signal = 1.0

        x2_clean = x2_clean * scale_signal
        x2_noisy = x2_clean + noise_t_unit

        if i < 3:
            tc1, z1 = match_filter_time_series(h1, x2_noisy, dt)
            snr1 = np.max(np.abs(z1))
            print(f"  → Zero‐noise SNR = {snr0:.2f}, post‐noise SNR ≃ {snr1:.2f}")
    else:
        x2_noisy = x2_clean

    # ─────────────────────────────────────────────────────────────────────────
    # (F.8) MATCH‐FILTER: find t_hat and φ_hat
    # ─────────────────────────────────────────────────────────────────────────
    tc, z_t_complex = match_filter_time_series(h1, x2_noisy, dt)
    idx_peak = np.argmax(np.abs(z_t_complex))
    raw_lag  = tc[idx_peak]
    t_hat    = raw_lag % duration
    phi_hat  = np.angle(z_t_complex[idx_peak])  # now returns a “truly complex” phase

    # ─────────────────────────────────────────────────────────────────────────
    # (F.9) APPLY ANALYTIC CORRECTIONS
    # ─────────────────────────────────────────────────────────────────────────
    t_hat_corr1  = (t_hat - dt1)               % duration
    t_hat_corr12 = (t_hat - (dt1 + dt2))       % duration

    phi_hat_corr1  = (phi_hat - dphi1)         % (2.0 * np.pi)
    phi_hat_corr12 = (phi_hat - (dphi1 + dphi2))% (2.0 * np.pi)

    # Debug: plot the first few injections
    if i < 3:
        print(f"\nInjection {i}:")
        print(f"  Template masses = ({m1_template:.1f}, {m2_template:.1f})   Injection masses = ({m1_inj:.1f}, {m2_inj:.1f})")
        print(f"  dt1={dt1:.3e}, dt2={dt2:.3e},   dphi1={dphi1:.3e}, dphi2={dphi2:.3e}")
        print(f"  t_true={t_true:.3f},   t_hat_raw={t_hat:.3f},   t_hat_corr1={t_hat_corr1:.3f},   t_hat_corr12={t_hat_corr12:.3f}")
        print(f"  φ_true={phi_true:.3f},   φ_hat_raw={phi_hat:.3f},   φ_hat_corr1={phi_hat_corr1:.3f},   φ_hat_corr12={phi_hat_corr12:.3f}")

        plt.figure(figsize=(10, 4))
        plt.plot(tc, np.abs(z_t_complex), label="SNR", color='blue')
        plt.axvline(raw_lag, color='red',    linestyle='--', label="peak lag")
        plt.axvline(t_hat,   color='green',  linestyle='--', label="t_hat")
        plt.xlabel("Time [s]"); plt.ylabel("SNR"); plt.title(f"Injection {i}: SNR Time Series")
        plt.legend(); plt.grid(); plt.show()

        plt.figure(figsize=(10, 4))
        times = np.arange(N) * dt
        # x2_clean and x2_noisy are real‐valued
        plt.plot(times, x2_clean, label="Clean injection", color='blue')
        plt.plot(times, x2_noisy, label="Noisy injection", color='orange', alpha=0.3)
        plt.axvline(t_true, color='red', linestyle='--', label="True t_c")
        plt.xlabel("Time [s]"); plt.ylabel("Amplitude");
        plt.title(f"Injection {i}: Clean vs Noisy Time Series")
        plt.legend(); plt.grid(); plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # (F.10) STORE FOR FINAL PLOTTING
    # ─────────────────────────────────────────────────────────────────────────
    t_true_list.append(t_true)
    t_hat_raw_list.append(t_hat)
    t_hat_corr1_list.append(t_hat_corr1)
    t_hat_corr12_list.append(t_hat_corr12)

    phi_true_list.append(phi_true)
    phi_hat_raw_list.append(phi_hat)
    phi_hat_corr1_list.append(phi_hat_corr1)
    phi_hat_corr12_list.append(phi_hat_corr12)


# ──────────────────────────────────────────────────────────────────────────────
# (G) CONVERT LISTS TO ARRAYS & COMPUTE WRAPPED RESIDUALS
# ──────────────────────────────────────────────────────────────────────────────

t_true_arr       = np.array(t_true_list)
t_hat_raw_arr    = np.array(t_hat_raw_list)
t_hat_corr1_arr  = np.array(t_hat_corr1_list)
t_hat_corr12_arr = np.array(t_hat_corr12_list)

phi_true_arr       = np.array(phi_true_list)
phi_hat_raw_arr    = np.array(phi_hat_raw_list)
phi_hat_corr1_arr  = np.array(phi_hat_corr1_list)
phi_hat_corr12_arr = np.array(phi_hat_corr12_list)

raw_time_res    = wrap_time_residuals(t_hat_raw_arr,   t_true_arr, duration)
corr1_time_res  = wrap_time_residuals(t_hat_corr1_arr, t_true_arr, duration)
corr12_time_res = wrap_time_residuals(t_hat_corr12_arr, t_true_arr, duration)

raw_phase_res    = wrap_phase_residuals(phi_hat_raw_arr,   phi_true_arr)
corr1_phase_res  = wrap_phase_residuals(phi_hat_corr1_arr, phi_true_arr)
corr12_phase_res = wrap_phase_residuals(phi_hat_corr12_arr, phi_true_arr)

# ──────────────────────────────────────────────────────────────────────────────
# (H) PLOT HISTOGRAMS & SCATTER
# ──────────────────────────────────────────────────────────────────────────────

plot_residuals_histograms(
    raw_time_res,    corr1_time_res,    corr12_time_res,
    raw_phase_res,   corr1_phase_res,   corr12_phase_res,
)

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
