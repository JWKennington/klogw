#!/usr/bin/env python
"""
psd_mismatch_demo_v3.py

 – Demonstration of first‐ and second‐order timing/phase bias purely from
  PSD‐whitening mismatch (no mass mismatches).

 – “Single example” shows raw vs. ∆t₁ vs. ∆t₁+∆t₂ on one injection.

 – “Multiple injections” now hold (m₁=30,M⊙, m₂=30,M⊙) fixed, but randomize
  t_true, phi_true.  This forces the matched filter to see a *perfect‐match*
  waveform (aside from PSD drift), so the only shift in t̂c,phîc is from the
  PSD mismatch.  That is what our analytic ∆t₁/∆t₂/∆phi₁/∆phi₂ formulas were
  supposed to predict.

Dependencies:
  • numpy
  • scipy
  • matplotlib
  • lalsuite  (lal, lalsimulation)
"""

import numpy as np
import matplotlib.pyplot as plt

import lal
import lalsimulation as lalsim

# =============================================================================
# Global parameters and PSD definitions
# =============================================================================

fs = 4096.0         # sampling rate [Hz]
duration = 20.0     # time‐series length [s]
N = int(fs * duration)
dt = 1.0 / fs
df = 1.0 / duration
freqs = np.fft.rfftfreq(N, d=dt)  # length = N/2 + 1

def psd_aLIGO(f):
    """One‐sided aLIGO Zero‐Det, High‐Power design PSD."""
    return lalsim.SimNoisePSDaLIGOZeroDetHighPower(f)

# build PSD1 and PSD2 arrays
PSD1 = np.array([psd_aLIGO(f) for f in freqs])
# perturbation = 0.5 * np.exp(-0.5 * ((freqs - 150.0) / 50.0)**2)
# random perturbations at every frequency, uniformly distributed
perturbation = np.random.uniform(-1, 1, size=PSD1.shape)
# Smooth the perturbation a bit to avoid large jumps
perturbation = np.convolve(perturbation, np.ones(10)/10, mode='same')
PSD2 = PSD1 * (1.0 + 0.1 * perturbation)

# replace any NaN or ≤0 by +∞ (especially at f=0)
PSD1[np.isnan(PSD1)] = np.inf
PSD2[np.isnan(PSD2)] = np.inf
PSD1[PSD1 <= 0.0] = np.inf
PSD2[PSD2 <= 0.0] = np.inf

# =============================================================================
# Build minimum‐phase phi_mp(f) from PSD2
# =============================================================================

def compute_minimum_phase_phi(psd, freqs):
    """
    Build the minimum‐phase whitening phase phi_mp(f) from a one‐sided PSD array psd[freqs].
    We force psd2[0] = psd2[1] so that log(psd2) is finite at DC.
    """
    M = len(freqs)
    # copy PSD, set f=0 to PSD[f=1], so log is finite
    psd2 = psd.copy()
    psd2[0] = psd2[1]

    # now logA_pos = -½ log(psd2) is finite everywhere
    logA_pos = -0.5 * np.log(psd2)

    # form full length‐N log spectrum (Hermitian symmetric)
    logA_full = np.zeros(N, dtype=np.float64)
    logA_full[:M] = logA_pos
    logA_full[M:] = logA_pos[-2:0:-1]

    # real cepstrum
    cepstrum = np.fft.ifft(logA_full).real

    # build min‐phase cepstrum: index0 stays, indices 1..(N/2−1) doubled,
    # index N/2 (if even) copied unmodified
    minphase_cepstrum = np.zeros_like(cepstrum)
    minphase_cepstrum[0] = cepstrum[0]
    minphase_cepstrum[1 : N // 2] = 2.0 * cepstrum[1 : N // 2]
    if (N % 2) == 0:
        minphase_cepstrum[N // 2] = cepstrum[N // 2]

    # back to log‐spectrum
    logMin_full = np.fft.fft(minphase_cepstrum)

    # phi_mp(f) = arg of positive half
    phi_mp = np.angle(logMin_full[:M])
    return phi_mp

phi_mp = compute_minimum_phase_phi(PSD2, freqs)

# whitening filters:
W1 = 1.0 / np.sqrt(PSD1)                # linear‐phase (zero‐phase) template‐whitening
W2 = (1.0 / np.sqrt(PSD2)) * np.exp(1j * phi_mp)  # minimum‐phase data‐whitening

# =============================================================================
# Generate a frequency‐domain TaylorF2 inspiral H0(f) for (m₁,m₂)=30,30 M⊙
# =============================================================================

def generate_fd_waveform(f_min, f_max, df, m1, m2, distance_mpc):
    """
    Return (freqs_in, H0) for TaylorF2 inspiral: m1,m2 in M⊙, distance in Mpc.
    H0 is a complex array on [f_min : f_max] spaced by df.
    """
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

    # if a list is returned, take the first (h+)
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

# “base” masses and distance (we will never change these in the injections)
m1_base, m2_base = 30.0, 30.0
distance_base = 500.0  # Mpc

f_min = freqs[1]
f_max = freqs[-1]
freqs_fd, H0_fd = generate_fd_waveform(f_min, f_max, df, m1_base, m2_base, distance_base)

# interpolate H0_fd onto our full “freqs” grid
H0 = np.zeros_like(freqs, dtype=complex)
H0[1:] = np.interp(freqs[1:], freqs_fd, H0_fd[1:], left=0.0, right=0.0)
H0[0] = 0.0

# =============================================================================
# Analytic correction formulas (first & second order)
# =============================================================================

def compute_first_order_corrections(H0, PSD2, phi_mp, freqs):
    """
    delta_t₁ = [∫ f |H0|²/PSD2 · phi_mp df] / [2π ∫ f² |H0|²/PSD2 df]
    delta_phi₁ = [∫ |H0|²/PSD2 · phi_mp df] / [∫ |H0|²/PSD2 df]
    """
    Wf = (np.abs(H0)**2) / PSD2
    fpos = freqs[1:]
    Wfpos = Wf[1:]
    Phi_pos = phi_mp[1:]

    num_t = np.trapz(fpos * Wfpos * Phi_pos, x=fpos)
    den_t = 2.0 * np.pi * np.trapz(fpos**2 * Wfpos, x=fpos)
    Delta_t1 = num_t / den_t

    num_phi = np.trapz(Wfpos * Phi_pos, x=fpos)
    den_phi = np.trapz(Wfpos, x=fpos)
    Delta_phi1 = num_phi / den_phi

    return Delta_t1, Delta_phi1

def compute_second_order_corrections(H0, PSD2, phi_mp, freqs, delta_t1, delta_phi1):
    """
    Approximate leading second‐order pieces (phi_mp²‐terms) for delta_t₂, delta_phi₂:
      delta_t₂ ≈ − [ ∫ f² W(f) phi_mp² df ] / [ (2π)² ∫ f² W(f) df ]
      delta_phi₂ ≈ − [ ∫ W(f) phi_mp² df ] / [ 2 ∫ W(f) df ]
    (Higher‐order mixed terms in delta_t1,delta_phi1 exist but are smaller.)
    """
    Wf = (np.abs(H0)**2) / PSD2
    fpos = freqs[1:]
    Wfpos = Wf[1:]
    Phi_pos = phi_mp[1:]

    num2_t = np.trapz((fpos**2) * Wfpos * (Phi_pos**2), x=fpos)
    den2_t = (2.0 * np.pi)**2 * np.trapz((fpos**2) * Wfpos, x=fpos)
    delta_t2 = - num2_t / den2_t

    num2_phi = np.trapz(Wfpos * (Phi_pos**2), x=fpos)
    den2_phi = 2.0 * np.trapz(Wfpos, x=fpos)
    delta_phi2 = - num2_phi / den2_phi

    return delta_t2, delta_phi2

# compute corrections for the “base” 30+30 M⊙ waveform:
delta_t1_base, delta_phi1_base = compute_first_order_corrections(H0, PSD2, phi_mp, freqs)
delta_t2_base, delta_phi2_base = compute_second_order_corrections(H0, PSD2, phi_mp, freqs, delta_t1_base, delta_phi1_base)

# =============================================================================
# SINGLE EXAMPLE: build one injection at (t_true=0, phi_true=0) just to show vertical lines
# =============================================================================

# whiten template & data for base waveform (no time/phase shift)
H1_base = H0 * W1
h1_base = np.fft.irfft(H1_base, n=N)
X2_base = H0 * W2
x2_base = np.fft.irfft(X2_base, n=N)

# matched‐filter in time domain
def match_filter_time_series(h1, x2):
    H1_t = np.fft.rfft(h1, n=N)
    X2_t = np.fft.rfft(x2, n=N)
    corr_f = np.conj(H1_t) * X2_t
    z_t = np.fft.irfft(corr_f, n=N)
    tc = np.fft.fftfreq(N, d=dt)
    z_t = np.fft.fftshift(z_t)
    tc = np.fft.fftshift(tc)
    return tc, z_t

tc_base, z_t_base = match_filter_time_series(h1_base, x2_base)
idx_peak_base = np.argmax(np.abs(z_t_base))
t_hat_raw_base = tc_base[idx_peak_base] % duration
phi_hat_raw_base = np.angle(z_t_base[idx_peak_base])

# apply corrections to that one peak
t_hat_corr1_base = (t_hat_raw_base - delta_t1_base) % duration
t_hat_corr12_base = (t_hat_raw_base - (delta_t1_base + delta_t2_base)) % duration

phi_hat_corr1_base = (phi_hat_raw_base - delta_phi1_base) % (2.0 * np.pi)
phi_hat_corr12_base = (phi_hat_raw_base - (delta_phi1_base + delta_phi2_base)) % (2.0 * np.pi)

# =============================================================================
# PLOT: single‐example matched‐filter output (vertical lines)
# =============================================================================

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.loglog(freqs[1:], PSD1[1:], label='PSD1 (linear‐phase)')
plt.loglog(freqs[1:], PSD2[1:], label='PSD2 (min‐phase)')
plt.xlim(10, fs/2)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [1/Hz]')
plt.legend()
plt.title('PSD1 vs. PSD2')

plt.subplot(3, 1, 2)
plt.plot(freqs[1:], phi_mp[1:], 'g-')
plt.xlim(10, fs/2)
plt.xlabel('frequency [Hz]')
plt.ylabel(r'$\phi_{\rm mp}(f)$ [rad]')
plt.title('Minimum‐Phase Filter Phase Difference')

plt.subplot(3, 1, 3)
t_arr = np.linspace(0, duration, N, endpoint=False)
center = N//2
window = 256
plt.plot(
    t_arr[center-window : center+window],
    h1_base[center-window : center+window],
    'b-', label='whitened template'
)
plt.plot(
    t_arr[center-window : center+window],
    x2_base[center-window : center+window],
    'r-', label='whitened data',
    alpha=0.7
)
plt.xlabel('time [s]')
plt.ylabel('strain (arb)')
plt.title('Zoomed Whitened Template vs. Whitened Data')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(tc_base, np.abs(z_t_base), 'k-')
_vline_min = np.min(np.abs(z_t_base))
_vline_max = np.max(np.abs(z_t_base))
plt.axvline(
    t_hat_raw_base,
    ymin=_vline_min, ymax=_vline_max,
    color='r',
    linestyle='--',
    label=f'raw $t_{{hat}}$ = {t_hat_raw_base:.4f}s',
)
plt.axvline(t_hat_corr1_base,
            ymin=_vline_min, ymax=_vline_max,
            color='b', linestyle='-.', label=f'1st‐order corr = {t_hat_corr1_base:.4f}s')
plt.axvline(t_hat_corr12_base,
            ymin=_vline_min, ymax=_vline_max,
            color='m', linestyle=':',  label=f'1+2nd‐order corr = {t_hat_corr12_base:.4f}s')
plt.xlabel('$t_c$ [s]')
plt.ylabel(r'$|Z(t_c)|$')
plt.title('Matched‐Filter Output: Single Example')
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# MULTIPLE INJECTIONS (fixed 30+30M⊙) → scatter raw vs. corrected t̂c, phîc
# =============================================================================

n_injections = 500
t_true_list       = []
t_hat_raw_list    = []
t_hat_corr1_list  = []
t_hat_corr12_list = []
phi_true_list       = []
phi_hat_raw_list    = []
phi_hat_corr1_list  = []
phi_hat_corr12_list = []

for _ in range(n_injections):
    # choose random true coalescence time and phase
    t_true = np.random.uniform(0.3, duration - 0.3)
    phi_true = np.random.uniform(0.0, 2.0 * np.pi)

    # use EXACTLY the same base H0(f) → template
    H0_i = H0.copy()  # same 30+30M⊙ waveform

    # compute that waveform’s delta_t₁, delta_phi₁, delta_t₂, delta_phi₂ (they are actually identical to base)
    Dt1_i, Dphi1_i = delta_t1_base, delta_phi1_base
    Dt2_i, Dphi2_i = delta_t2_base, delta_phi2_base

    # whitened template (in time) for injection:
    H1_i = H0_i * W1
    h1_i = np.fft.irfft(H1_i, n=N)

    # make the “injection in data”: shift a copy of H0_i by (t_true, phi_true)
    phase_shift = np.exp(-2j * np.pi * freqs * t_true)
    Hsig_i = H0_i * np.exp(1j * phi_true) * phase_shift
    X2_i = Hsig_i * W2
    x2_i = np.fft.irfft(X2_i, n=N)

    # matched filter between h1_i and x2_i
    tc_i, z_t_i = match_filter_time_series(h1_i, x2_i)
    idx_peak_i = np.argmax(np.abs(z_t_i))
    t_hat    = (tc_i[idx_peak_i] ) % duration
    phi_hat    = np.angle(z_t_i[idx_peak_i])

    # apply corrections
    t_hat_corr1  = (t_hat - Dt1_i)        % duration
    t_hat_corr12 = (t_hat - (Dt1_i + Dt2_i)) % duration

    phi_hat_corr1  = (phi_hat - Dphi1_i)        % (2.0*np.pi)
    phi_hat_corr12 = (phi_hat - (Dphi1_i + Dphi2_i)) % (2.0*np.pi)

    # store
    t_true_list.append(t_true)
    t_hat_raw_list.append(t_hat)
    t_hat_corr1_list.append(t_hat_corr1)
    t_hat_corr12_list.append(t_hat_corr12)

    phi_true_list.append(phi_true)
    phi_hat_raw_list.append(phi_hat)
    phi_hat_corr1_list.append(phi_hat_corr1)
    phi_hat_corr12_list.append(phi_hat_corr12)

# convert to arrays
t_true_arr       = np.array(t_true_list)
t_hat_raw_arr    = np.array(t_hat_raw_list)
t_hat_corr1_arr  = np.array(t_hat_corr1_list)
t_hat_corr12_arr = np.array(t_hat_corr12_list)

phi_true_arr       = np.array(phi_true_list)
phi_hat_raw_arr    = np.array(phi_hat_raw_list)
phi_hat_corr1_arr  = np.array(phi_hat_corr1_list)
phi_hat_corr12_arr = np.array(phi_hat_corr12_list)

# =============================================================================
# Plot scatter: raw vs. corrected
# =============================================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(t_true_arr,   t_hat_raw_arr,    color='red',   alpha=0.5, label='Raw  $t_{hat}$')
plt.scatter(t_true_arr,   t_hat_corr1_arr,  color='blue',  alpha=0.5, label='1st‐order corr')
plt.scatter(t_true_arr,   t_hat_corr12_arr, color='magenta',alpha=0.5, label='1+2nd‐order corr')
plt.plot([0, duration],[0, duration],'k--', label='Ideal')
plt.xlabel('True $t_c$ [s]')
plt.ylabel('Estimated $t_c$ [s]')
plt.title('Coalescence Time: Raw vs. Corrections')
plt.legend()
plt.xlim(0, duration)
plt.ylim(0, duration)

plt.subplot(1,2,2)
plt.scatter(phi_true_arr,    phi_hat_raw_arr,    color='red',   alpha=0.5, label='Raw  $phi_{hat}$')
plt.scatter(phi_true_arr,    phi_hat_corr1_arr,  color='blue',  alpha=0.5, label='1st‐order corr')
plt.scatter(phi_true_arr,    phi_hat_corr12_arr, color='magenta',alpha=0.5, label='1+2nd‐order corr')
plt.plot([0,2*np.pi],[0,2*np.pi],'k--', label='Ideal')
plt.xlabel('True $phi_c$ [rad]')
plt.ylabel('Estimated $phi_c$ [rad]')
plt.title('Coalescence Phase: Raw vs. Corrections')
plt.legend()
plt.xlim(0, 2*np.pi)
plt.ylim(0, 2*np.pi)

plt.tight_layout()
plt.show()
