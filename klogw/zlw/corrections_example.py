#!/usr/bin/env python
"""
psd_mismatch_demo_updated2.py

Demonstration of first- and second-order timing and phase biases due to PSD whitening mismatch.
This updated version fixes the interpolation lengths for H0_fd vs. freqs_fd.

Dependencies:
    - numpy
    - scipy
    - matplotlib
    - lalsuite (lal, lalsimulation)
"""

import numpy as np
import matplotlib.pyplot as plt

import lal
import lalsimulation as lalsim

# =============================================================================
# Parameters and PSD definitions
# =============================================================================

fs = 4096          # Sampling rate (Hz)
duration = 2.0     # Duration of time series (seconds)
N = int(fs * duration)
dt = 1.0 / fs
df = 1.0 / duration
freqs = np.fft.rfftfreq(N, d=dt)  # Length N/2 + 1

def psd_aLIGO(f):
    """aLIGO Zero Det, High Power design PSD (one-sided)."""
    return lalsim.SimNoisePSDaLIGOZeroDetHighPower(f)

PSD1 = np.array([psd_aLIGO(f) for f in freqs])
perturbation = 0.1 * np.exp(-0.5 * ((freqs - 150.0) / 50.0)**2)
PSD2 = PSD1 * (1.0 + perturbation)
PSD1[PSD1 <= 0] = np.inf
PSD2[PSD2 <= 0] = np.inf

# =============================================================================
# Minimum-phase filter computation
# =============================================================================

def compute_minimum_phase_phi(psd, freqs):
    M = len(freqs)
    logA_pos = -0.5 * np.log(psd)
    logA_full = np.zeros(N, dtype=np.float64)
    logA_full[:M] = logA_pos
    logA_full[M:] = logA_pos[-2:0:-1]
    cepstrum = np.fft.ifft(logA_full).real
    minphase_cepstrum = np.zeros_like(cepstrum)
    minphase_cepstrum[0] = cepstrum[0]
    minphase_cepstrum[1:N//2] = 2.0 * cepstrum[1:N//2]
    if N % 2 == 0:
        minphase_cepstrum[N//2] = cepstrum[N//2]
    logMin_full = np.fft.fft(minphase_cepstrum)
    phi_mp = np.angle(logMin_full[:M])
    return phi_mp

phi_mp = compute_minimum_phase_phi(PSD2, freqs)

# =============================================================================
# Whitening filters
# =============================================================================

W1 = 1.0 / np.sqrt(PSD1)
W2 = (1.0 / np.sqrt(PSD2)) * np.exp(1j * phi_mp)

# =============================================================================
# Generate frequency-domain Inspiral waveform
# =============================================================================

def generate_fd_waveform(f_min, f_max, df, m1, m2, distance_mpc):
    lal_msun = lal.MSUN_SI
    m1_SI = m1 * lal_msun
    m2_SI = m2 * lal_msun
    freqs_in = np.arange(f_min, f_max + df, df)
    dist_SI = distance_mpc * 1e6 * lal.PC_SI
    spin1x = spin1y = spin1z = 0.0
    spin2x = spin2y = spin2z = 0.0
    eccentricity = 0.0
    meanPerAno = 0.0
    incl = 0.0
    phiRef = 0.0
    longAscNodes = 0.0
    fRef = freqs_in[0]

    Hf_raw = lalsim.SimInspiralFD(
        m1_SI, m2_SI,
        spin1x, spin1y, spin1z,
        spin2x, spin2y, spin2z,
        dist_SI,
        incl, phiRef, longAscNodes,
        eccentricity, meanPerAno,
        df, f_min, f_max, fRef,
        lal.CreateDict(),
        lalsim.TaylorF2
    )

    if isinstance(Hf_raw, (list, tuple)):
        Hf_series = Hf_raw[0]
    else:
        Hf_series = Hf_raw

    length = Hf_series.data.length
    H0 = np.zeros(length, dtype=np.complex128)
    for idx in range(length):
        comp = Hf_series.data.data[idx]
        H0[idx] = complex(comp.real, comp.imag)

    return freqs_in, H0

m1, m2 = 30.0, 30.0
distance_mpc = 500.0
f_min = freqs[1]
f_max = freqs[-1]
freqs_fd, H0_fd = generate_fd_waveform(f_min, f_max, df, m1, m2, distance_mpc)

# =============================================================================
# Interpolate H0_fd onto full freqs grid
# =============================================================================

# freqs_fd length is len(H0_fd)-1, because H0_fd includes DC at index 0.
# Drop H0_fd[0] to align lengths.
H0 = np.zeros_like(freqs, dtype=complex)
H0[1:] = np.interp(freqs[1:], freqs_fd, H0_fd[1:], left=0, right=0)
H0[0] = 0.0

# =============================================================================
# Whitening and inverse FFT
# =============================================================================

H1 = H0 * W1
h1 = np.fft.irfft(H1, n=N)
X2 = H0 * W2
x2 = np.fft.irfft(X2, n=N)

# =============================================================================
# Matched filtering
# =============================================================================

def match_filter_time_series(h1, x2):
    H1_t = np.fft.rfft(h1, n=N)
    X2_t = np.fft.rfft(x2, n=N)
    corr_f = np.conj(H1_t) * X2_t
    z_t = np.fft.irfft(corr_f, n=N)
    tc = np.fft.fftfreq(N, d=dt)
    z_t = np.fft.fftshift(z_t)
    tc = np.fft.fftshift(tc)
    return tc, z_t

tc, z_t = match_filter_time_series(h1, x2)
idx_peak = np.argmax(np.abs(z_t))
t_hat_raw = tc[idx_peak] % duration
phi_hat_raw = np.angle(z_t[idx_peak])

# =============================================================================
# Analytic first-order corrections
# =============================================================================

Wf = (np.abs(H0)**2) / PSD2
fpos = freqs[1:]
Wfpos = Wf[1:]
Phi = phi_mp
Phi_pos = Phi[1:]

num_t = np.trapz(fpos * Wfpos * Phi_pos, x=fpos)
den_t = 2 * np.pi * np.trapz(fpos**2 * Wfpos, x=fpos)
Delta_t1 = num_t / den_t

num_phi = np.trapz(Wfpos * Phi_pos, x=fpos)
den_phi = np.trapz(Wfpos, x=fpos)
Delta_phi1 = num_phi / den_phi

# =============================================================================
# Plotting single example
# =============================================================================

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.loglog(freqs[1:], PSD1[1:], label='PSD1 (linear-phase)')
plt.loglog(freqs[1:], PSD2[1:], label='PSD2 (min-phase)')
plt.xlim(10, fs/2)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [1/Hz]')
plt.legend()
plt.title('PSD1 vs PSD2')

plt.subplot(3, 1, 2)
plt.plot(freqs[1:], phi_mp[1:], 'g-')
plt.xlim(10, fs/2)
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'$\phi_{\rm mp}(f)$ [rad]')
plt.title('Minimum-Phase Filter Phase Difference')

plt.subplot(3, 1, 3)
t = np.linspace(0, duration, N, endpoint=False)
center = N // 2
window = 256
plt.plot(t[center-window:center+window], h1[center-window:center+window], 'b-', label='Whitened template')
plt.plot(t[center-window:center+window], x2[center-window:center+window], 'r-', label='Whitened data', alpha=0.7)
plt.xlabel('Time [s]')
plt.ylabel('Strain (arb)')
plt.title('Zoomed Whitened Template vs Whitened Data')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(tc, np.abs(z_t), 'k-')
plt.axvline(t_hat_raw, color='r', linestyle='--', label=f'Raw $t_{{\\mathrm{{hat}}}}$ = {t_hat_raw:.4f}s')
plt.xlabel('$t_c$ [s]')
plt.ylabel(r'$|Z(t_c)|$')
plt.title('Matched-Filter Output')
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# Multiple injections demonstration
# =============================================================================

n_injections = 50
t_true_list = []
t_hat_list = []
t_hat_corr_list = []
phi_true_list = []
phi_hat_list = []
phi_hat_corr_list = []

for _ in range(n_injections):
    t_true = np.random.uniform(0.3, 1.7)
    phi_true = np.random.uniform(0, 2 * np.pi)

    phase_shift = np.exp(-2j * np.pi * freqs * t_true)
    H_sig = H0 * np.exp(1j * phi_true) * phase_shift

    X2_i = H_sig * W2
    x2_i = np.fft.irfft(X2_i, n=N)

    H1_i = H0 * W1
    h1_i = np.fft.irfft(H1_i, n=N)

    tc_i, z_t_i = match_filter_time_series(h1_i, x2_i)
    idx_peak_i = np.argmax(np.abs(z_t_i))
    t_hat = tc_i[idx_peak_i] % duration
    phi_hat = np.angle(z_t_i[idx_peak_i])

    t_hat_corr = (t_hat - Delta_t1) % duration
    phi_hat_corr = (phi_hat - Delta_phi1) % (2 * np.pi)

    t_true_list.append(t_true)
    t_hat_list.append(t_hat)
    t_hat_corr_list.append(t_hat_corr)
    phi_true_list.append(phi_true)
    phi_hat_list.append(phi_hat)
    phi_hat_corr_list.append(phi_hat_corr)

t_true_arr = np.array(t_true_list)
t_hat_arr = np.array(t_hat_list)
t_hat_corr_arr = np.array(t_hat_corr_list)
phi_true_arr = np.array(phi_true_list)
phi_hat_arr = np.array(phi_hat_list)
phi_hat_corr_arr = np.array(phi_hat_corr_list)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(t_true_arr, t_hat_arr, color='r', alpha=0.7, label='Raw $t_{\\mathrm{hat}}$')
plt.scatter(t_true_arr, t_hat_corr_arr, color='b', alpha=0.7, label='Corrected $t_{\\mathrm{hat}}$')
plt.plot([0, 2], [0, 2], 'k--', label='Ideal')
plt.xlabel('True $t_c$ [s]')
plt.ylabel('Estimated $t_c$ [s]')
plt.title('Coalescence Time: Raw vs. Corrected')
plt.legend()
plt.xlim(0, 2)
plt.ylim(0, 2)

plt.subplot(1, 2, 2)
plt.scatter(phi_true_arr, phi_hat_arr, color='r', alpha=0.7, label='Raw $\\phi_{\\mathrm{hat}}$')
plt.scatter(phi_true_arr, phi_hat_corr_arr, color='b', alpha=0.7, label='Corrected $\\phi_{\\mathrm{hat}}$')
plt.plot([0, 2*np.pi], [0, 2*np.pi], 'k--', label='Ideal')
plt.xlabel('True $\\phi_c$ [rad]')
plt.ylabel('Estimated $\\phi_c$ [rad]')
plt.title('Coalescence Phase: Raw vs. Corrected')
plt.legend()
plt.xlim(0, 2*np.pi)
plt.ylim(0, 2*np.pi)

plt.tight_layout()
plt.show()
