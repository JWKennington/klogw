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

Dependencies (same as before):
  • numpy
  • matplotlib
  • lalsuite  (lal, lalsimulation)
  • zlw_utils.py (the helper functions we wrote earlier)
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

VARY_PSD                = True    # If True, each injection may use a different PSD2
VARY_TEMPLATE_INTRINSICS = True    # If True, pick a random (m1,m2) for the *template*
VARY_INJ_INTRINSICS      = True    # If True, injection masses = template masses + small offset
ADD_PSD_SHAPED_NOISE     = True    # If True, add whitened Gaussian noise matched to PSD2

# ──────────────────────────────────────────────────────────────────────────────
# (B) FIXED RUN PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

fs       = 16384.0            # Sampling frequency [Hz]
duration = 20.0               # Duration [s]
N        = int(fs * duration)
dt       = 1.0 / fs
df       = 1.0 / duration

# Build the one‐sided frequency axis of length (N//2 + 1)
freqs = np.fft.rfftfreq(N, d=dt)

# “Base” template‐intrinsic ranges (only used if VARY_TEMPLATE_INTRINSICS=True)
TEMPLATE_M1_RANGE = (10.0, 100.0)  # M⊙
TEMPLATE_M2_RANGE = (10.0, 100.0)  # M⊙

# “Max injection‐intrinsic error” (only if VARY_INJ_INTRINSICS=True)
INJ_MASS_OFFSET_STD = 1.0   # ±1 M⊙ Gaussian scatter around the template’s (m1,m2)

# Choose how many total injections you want to do:
n_injections = 500

# These are used if VARY_PSD=False (i.e. fixed PSD2):
PSD2_FIXED_KIND       = "gaussian"
PSD2_FIXED_KWARGS     = {"center": 150.0, "width": 50.0, "amplitude_ratio": 0.3}

# If VARY_PSD=True, we will build a *new* PSD2 per injection by sampling
# from one of these "perturbation configurations":
PERTURBATION_LIST = [
    {"kind": "gaussian", "center":  50.0,  "width":  20.0, "amplitude_ratio": 0.3},
    {"kind": "gaussian", "center": 100.0,  "width":  50.0, "amplitude_ratio": 0.3},
    {"kind": "gaussian", "center": 200.0,  "width": 100.0, "amplitude_ratio": 0.3},
    {"kind": "random",   "amplitude_range": 0.3},
]

# Base “template” masses (only used if VARY_TEMPLATE_INTRINSICS=False)
BASE_M1 = 30.0  # M⊙
BASE_M2 = 30.0  # M⊙

# Base “injection” distance (affects SNR).  You can increase to reduce SNR.
BASE_DISTANCE_MPC = 100.0


# ──────────────────────────────────────────────────────────────────────────────
# (C) A HELPER TO GENERATE “PSD-SHAPED” WHITENED NOISE
# ──────────────────────────────────────────────────────────────────────────────

def make_whitened_noise(PSD2, dt):
    """
    Return one realization of a *whitened*, zero-mean Gaussian noise time series
    of length N, given the one‐sided PSD2[freqs].  The output has the property that
    when you do an FFT of it and multiply by sqrt(PSD2), you recover a white spectrum.

    Steps:
      1) Draw independent real Gaussians for Re and Im at each positive‐frequency bin.
      2) Combine them as (Re + i Im)/sqrt(2 df) so that the power per bin is 1/df.
      3) Multiply by 1/sqrt(PSD2) to “whiten” them.
      4) Mirror to negative freqs and IFFT → a real time series.
    """
    M  = PSD2.shape[0]           # = N//2 + 1
    N_full = (M - 1) * 2         # recover the full-length
    df_full = 1.0 / (N_full * dt)

    # (1) Draw Gaussians for each bin of the single‐sided spectrum:
    re = np.random.normal(size=M)
    im = np.random.normal(size=M)

    # (2) Combine into one‐sided complex with unit variance per bin:
    noise_fd_pos = (re + 1j * im) / np.sqrt(2.0 * df_full)

    # (3) Whiten by PSD2:
    noise_fd_pos *= 1.0 / np.sqrt(PSD2)

    # (4) Build the full Hermitian symmetric spectrum of length N_full:
    noise_fd_full = np.zeros(N_full, dtype=complex)
    noise_fd_full[:M] = noise_fd_pos
    noise_fd_full[M:] = np.conj(noise_fd_pos[-2:0:-1])

    # (5) Inverse FFT → real time series:
    noise_t = np.fft.ifft(noise_fd_full, n=N_full)
    return noise_t.real


# ──────────────────────────────────────────────────────────────────────────────
# (D) LOAD THE “DESIGN” PSD1 ON freqs
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
    # -----------------------------------
    # (F.1) DRAW RANDOM COALESCENCE TIME & PHASE
    # -----------------------------------
    t_true   = np.random.uniform(0.3, duration - 0.3)
    phi_true = np.random.uniform(0.0, 2.0 * np.pi)

    # -----------------------------------
    # (F.2) POSSIBLY VARY THE PSD (PSD2)
    # -----------------------------------
    if VARY_PSD:
        # Pick one of the predefined perturbation recipes at random:
        cfg = PERTURBATION_LIST[np.random.randint(len(PERTURBATION_LIST))]
        PSD2 = make_perturbed_psd(PSD1, freqs, **cfg)
    else:
        PSD2 = make_perturbed_psd(PSD1, freqs, **PSD2_FIXED_KIND, **PSD2_FIXED_KWARGS)

    # Build the minimum‐phase φ_mp for this PSD2:
    phi_mp = compute_minimum_phase_phi(PSD2, freqs, N)

    # Build the whitening filters for this PSD1→PSD2:
    W1, W2 = make_whitening_filters(PSD1, PSD2, phi_mp)

    # -----------------------------------
    # (F.3) CHOOSE TEMPLATE INTRINSICS (m1_template, m2_template)
    # -----------------------------------
    if VARY_TEMPLATE_INTRINSICS:
        m1_template = np.random.uniform(*TEMPLATE_M1_RANGE)
        m2_template = np.random.uniform(*TEMPLATE_M2_RANGE)
    else:
        m1_template = BASE_M1
        m2_template = BASE_M2

    # Generate the *template* waveform H0_template(fd):
    freqs_fd_temp, H0_fd_temp = generate_fd_waveform(
        freqs[1], freqs[-1], df, m1_template, m2_template, BASE_DISTANCE_MPC
    )
    H0_template = interpolate_H0(freqs, freqs_fd_temp, H0_fd_temp)

    # -----------------------------------
    # (F.4) CHOOSE INJECTION INTRINSICS (m1_inj, m2_inj)
    # -----------------------------------
    if VARY_INJ_INTRINSICS:
        # take the template masses, add a small Gaussian error:
        m1_inj = m1_template + np.random.normal(scale=INJ_MASS_OFFSET_STD)
        m2_inj = m2_template + np.random.normal(scale=INJ_MASS_OFFSET_STD)
    else:
        m1_inj = m1_template
        m2_inj = m2_template

    # Make sure the injection masses are ≥ 1 M⊙ (just in case):
    m1_inj = max(1.0, m1_inj)
    m2_inj = max(1.0, m2_inj)

    # Generate the *injection* waveform H0_inj(fd):
    freqs_fd_inj, H0_fd_inj = generate_fd_waveform(
        freqs[1], freqs[-1], df, m1_inj, m2_inj, BASE_DISTANCE_MPC
    )
    H0_inj = interpolate_H0(freqs, freqs_fd_inj, H0_fd_inj)

    # -----------------------------------
    # (F.5) COMPUTE ANALYTIC CORRECTIONS BASED ON THE *TEMPLATE* ONLY
    # -----------------------------------
    dt1, dphi1, dt2, dphi2 = compute_corrections(H0_template, PSD2, phi_mp, freqs)

    # -----------------------------------
    # (F.6) BUILD THE “WHITENED TEMPLATE” TIME SERIES (complex, no roll)
    # -----------------------------------
    H1 = H0_template * W1       # length‐M = N//2+1 frequency‐domain
    h1 = np.fft.ifft(H1, n=N)   # length‐N complex time series

    # -----------------------------------
    # (F.7) BUILD THE “WHITENED INJECTION” TIME SERIES (real, no roll in make_injection_time_series)
    # -----------------------------------
    # (make_injection_time_series already does: H_sig(f)=H0_inj·e^(i φ_true)·e^(−2π i f t_true),
    #  then X2(f)=H_sig(f)·W2(f), then x2(t)=irfft, *no roll*.)
    x2_clean = make_injection_time_series(H0_inj, W2, freqs, N, dt, t_true, phi_true)

    # Add PSD‐shaped Gaussian noise around zero:
    if ADD_PSD_SHAPED_NOISE:
        # (1) Build a fresh whitened‐noise realization of length N:
        noise_t_raw = make_whitened_noise(PSD2, dt)  # RMS matched‐filter = 1

        # (2) Measure the noise‐free SNR of this injection:
        tc0, z0 = match_filter_time_series(h1, x2_clean, dt)
        snr0 = np.max(np.abs(z0))  # e.g. ∼20–30 for (30+30)@100 Mpc

        # (3) Choose a reasonable “target SNR” (so phases/times aren’t totally lost)
        SNR_target = 10.0

        # (4) Compute how much to scale the noise so that
        #     “injection + noise” peaks at ~ SNR_target:
        alpha = snr0 / SNR_target

        # (5) Scale down the noise:
        noise_t_scaled = noise_t_raw / alpha

        # (6) Build the final noisy injection:
        x2_noisy = x2_clean + noise_t_scaled

        # Debug (first few injections):
        if i < 3:
            tc1, z1 = match_filter_time_series(h1, x2_noisy, dt)
            snr1 = np.max(np.abs(z1))
            print(f"  → Zero‐noise SNR = {snr0:.2f}, after noise SNR ≃ {snr1:.2f}")
    else:
        x2_noisy = x2_clean

    # -----------------------------------
    # (F.8) MATCH‐FILTER: find t_hat and phi_hat
    # -----------------------------------
    tc, z_t_complex = match_filter_time_series(h1, x2_noisy, dt)
    idx_peak = np.argmax(np.abs(z_t_complex))
    raw_lag  = tc[idx_peak]         # in [−T/2, +T/2)
    t_hat    = raw_lag % duration   # wrap into [0, T)
    phi_hat  = np.angle(z_t_complex[idx_peak])

    # -----------------------------------
    # (F.9) APPLY THE ANALYTIC CORRECTIONS
    # -----------------------------------
    t_hat_corr1  = (t_hat - dt1)                 % duration
    t_hat_corr12 = (t_hat - (dt1 + dt2))         % duration

    phi_hat_corr1  = (phi_hat - dphi1)           % (2.0 * np.pi)
    phi_hat_corr12 = (phi_hat - (dphi1 + dphi2)) % (2.0 * np.pi)

    # (Optional debug print on the first few injections)
    if i < 3:
        print(f"\nInjection {i}:")
        print(f"  Template masses   = ({m1_template:.1f}, {m2_template:.1f}) M⊙")
        print(f"  Injection masses  = ({m1_inj:.1f}, {m2_inj:.1f}) M⊙")
        print(
            f"  PSD perturbation  = {('varied' if VARY_PSD else 'fixed')}  →  dt1={dt1:.3e} s, dt2={dt2:.3e} s"
        )
        print(f"                    dphi1={dphi1:.3e} rad, dphi2={dphi2:.3e} rad")

        # Plot the snr timeseries
        plt.figure(figsize=(10, 4))
        plt.plot(tc, np.abs(z_t_complex), label="SNR", color='blue')
        plt.axvline(raw_lag, color='red', linestyle='--', label="Peak lag")
        plt.axvline(t_hat, color='green', linestyle='--', label="t_hat")
        plt.title(f"Injection {i}: SNR Time Series")
        plt.xlabel("Time [s]")
        plt.ylabel("SNR")
        plt.legend()
        plt.grid()
        plt.show()

        # Plot the clean data
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(N) * dt, x2_clean, label="Clean injection", color='blue')
        plt.plot(np.arange(N) * dt, x2_noisy, label="Noisy injection",
                 color='orange', alpha=0.1)
        plt.axvline(t_true, color='red', linestyle='--', label="True time")
        plt.title(f"Injection {i}: Clean vs Noisy Time Series")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.show()

    # -----------------------------------
    # (F.10) STORE EVERYTHING FOR PLOTTING LATE
    # -----------------------------------
    t_true_list.append(t_true)
    t_hat_raw_list.append(t_hat)
    t_hat_corr1_list.append(t_hat_corr1)
    t_hat_corr12_list.append(t_hat_corr12)

    phi_true_list.append(phi_true)
    phi_hat_raw_list.append(phi_hat)
    phi_hat_corr1_list.append(phi_hat_corr1)
    phi_hat_corr12_list.append(phi_hat_corr12)


# ──────────────────────────────────────────────────────────────────────────────
# (G) CONVERT LISTS TO NUMPY ARRAYS AND COMPUTE “WRAPPED” RESIDUALS
# ──────────────────────────────────────────────────────────────────────────────

t_true_arr        = np.array(t_true_list)
t_hat_raw_arr     = np.array(t_hat_raw_list)
t_hat_corr1_arr   = np.array(t_hat_corr1_list)
t_hat_corr12_arr  = np.array(t_hat_corr12_list)

phi_true_arr        = np.array(phi_true_list)
phi_hat_raw_arr     = np.array(phi_hat_raw_list)
phi_hat_corr1_arr   = np.array(phi_hat_corr1_list)
phi_hat_corr12_arr  = np.array(phi_hat_corr12_list)

# Wrap timing residuals into [−T/2, +T/2):
raw_time_res    = wrap_time_residuals(t_hat_raw_arr,   t_true_arr, duration)
corr1_time_res  = wrap_time_residuals(t_hat_corr1_arr, t_true_arr, duration)
corr12_time_res = wrap_time_residuals(t_hat_corr12_arr, t_true_arr, duration)

# Wrap phase residuals into [−π, +π):
raw_phase_res    = wrap_phase_residuals(phi_hat_raw_arr,   phi_true_arr)
corr1_phase_res  = wrap_phase_residuals(phi_hat_corr1_arr, phi_true_arr)
corr12_phase_res = wrap_phase_residuals(phi_hat_corr12_arr, phi_true_arr)


# ──────────────────────────────────────────────────────────────────────────────
# (H) PLOT HISTOGRAMS OF RESIDUALS (TIMING & PHASE)
# ──────────────────────────────────────────────────────────────────────────────

plot_residuals_histograms(
    raw_time_res,    corr1_time_res,    corr12_time_res,
    raw_phase_res,   corr1_phase_res,   corr12_phase_res,
)

# ──────────────────────────────────────────────────────────────────────────────
# (I) PLOT SCATTER OF TRUE vs. ESTIMATED (TIMING & PHASE)
# ──────────────────────────────────────────────────────────────────────────────

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
