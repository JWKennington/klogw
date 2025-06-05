import numpy as np
import lal
import lalsimulation as lalsim
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1) PSD‐RELATED FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def load_design_psd(freqs):
    """
    Given an array of positive frequencies (one‐sided), return the aLIGO
    Zero‐Det, High‐Power design PSD evaluated at each freq.
    Any NaNs or non‐positive values at f=0 are replaced with +∞.
    """
    psd = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(f) for f in freqs])
    psd[np.isnan(psd)] = np.inf
    psd[psd <= 0.0] = np.inf
    return psd


def make_perturbed_psd(psd1, freqs, kind="gaussian", **kwargs):
    """
    Build a perturbed PSD2 from base PSD1 according to one of several 'kind's:

    - kind="gaussian":  kwargs should include
        center          = center frequency (Hz),
        width           = Gaussian sigma (Hz),
        amplitude_ratio = scale factor for fractional bump (e.g. 0.3 means PSD2=PSD1*(1+0.3*G)),
      i.e.
        G(f) = exp[-0.5*((f-center)/width)^2]
        PSD2 = PSD1 * (1 + amplitude_ratio * G(f)).

    - kind="random":  kwargs should include
        amplitude_range = maximum absolute fractional perturbation (e.g. 0.3),
      i.e.
        R(f) ~ Uniform[-amplitude_range, +amplitude_range]
        PSD2 = PSD1 * (1 + R(f)).

    - kind="custom":  kwargs should supply
        custom_array = same length as PSD1, representing fractional (ΔPSD/PSD1).
      i.e.
        PSD2 = PSD1 * (1 + custom_array).

    Returns a sanitized PSD2 (NaN or <=0 replaced with +∞).
    """
    if kind == "gaussian":
        center = kwargs.get("center", 200.0)
        width  = kwargs.get("width", 50.0)
        amplitude_ratio = kwargs.get("amplitude_ratio", 0.3)
        G = np.exp(-0.5 * ((freqs - center) / width) ** 2)
        psd2 = psd1 * (1.0 + amplitude_ratio * G)

    elif kind == "random":
        amp_range = kwargs.get("amplitude_range", 0.3)
        R = np.random.uniform(-amp_range, amp_range, size=psd1.shape)
        psd2 = psd1 * (1.0 + R)

    elif kind == "custom":
        custom_array = kwargs["custom_array"]
        if custom_array.shape != psd1.shape:
            raise ValueError("custom_array must have same shape as psd1")
        psd2 = psd1 * (1.0 + custom_array)

    else:
        raise ValueError(f"Unknown perturbation kind: {kind}")

    psd2 = psd2.copy()
    psd2[np.isnan(psd2)] = np.inf
    psd2[psd2 <= 0.0] = np.inf
    return psd2


def compute_minimum_phase_phi(psd, freqs, N):
    """
    Given a one‐sided PSD array “psd” on freqs[0..M-1], build the
    minimum‐phase whitening phase phi_mp[0..M-1] for an FFT length of N.

    Steps:
    1) Copy psd→psd2 and set psd2[0] = psd2[1] so log(psd2) is finite.
    2) logA_pos = -0.5 * log(psd2).
    3) Build a length‐N “logA_full” by Hermitian mirroring:
            logA_full[:M] = logA_pos
            logA_full[M:] = logA_pos[M-2 : 0 : -1]
    4) cepstrum = real( IFFT(logA_full) ).
    5) minphase_cepstrum[0] = cepstrum[0];
       minphase_cepstrum[1..N//2-1] = 2*cepstrum[1..N//2-1];
       if N even, minphase_cepstrum[N//2] = cepstrum[N//2].
    6) logMin_full = FFT(minphase_cepstrum).
    7) Return phi_mp = angle(logMin_full[:M]).
    """
    M = len(freqs)
    psd2 = psd.copy()
    psd2[0] = psd2[1]  # avoid -inf in log

    logA_pos = -0.5 * np.log(psd2)

    logA_full = np.zeros(N, dtype=np.float64)
    logA_full[:M] = logA_pos
    logA_full[M:] = logA_pos[-2:0:-1]

    cepstrum = np.fft.ifft(logA_full).real

    minp_cep = np.zeros_like(cepstrum)
    minp_cep[0] = cepstrum[0]
    minp_cep[1 : N//2] = 2.0 * cepstrum[1 : N//2]
    if (N % 2) == 0:
        minp_cep[N//2] = cepstrum[N//2]

    logMin_full = np.fft.fft(minp_cep)
    phi_mp = np.angle(logMin_full[:M])
    return phi_mp


def make_whitening_filters(PSD1, PSD2, phi_mp):
    """
    Given:
      – PSD1[0..M-1]: one‐sided PSD for template whitening (linear‐phase)
      – PSD2[0..M-1]: one‐sided PSD for data whitening (minimum‐phase)
      – phi_mp[0..M-1]: minimum‐phase spectral phase for whitening
    Return:
      – W1[0..M-1] = 1 / sqrt(PSD1)             (zero‐phase template whitening)
      – W2[0..M-1] = (1 / sqrt(PSD2)) e^{i phi_mp}  (min‐phase data whitening)
    """
    W1 = 1.0 / np.sqrt(PSD1)
    W2 = (1.0 / np.sqrt(PSD2)) * np.exp(1j * phi_mp)
    return W1, W2


# ──────────────────────────────────────────────────────────────────────────────
# 2) WAVEFORM‐RELATED FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def generate_fd_waveform(f_min, f_max, df, m1, m2, distance_mpc, approximant=lalsim.TaylorF2):
    """
    Generate a frequency‐domain inspiral waveform H0(f) for a non‐spinning
    TaylorF2 binary (m1,m2) [solar masses], at distance [Mpc], between f_min..f_max
    with frequency step df. Returns:
      – freqs_in:  array([f_min, f_min+df, ..., f_max])
      – H0_fd:     complex numpy array of length len(freqs_in).

    Uses lalsim.SimInspiralFD under the hood:
        SimInspiralFD(m1_SI, m2_SI, S1x, S1y, S1z, S2x, S2y, S2z,
                      distance_SI,
                      inclination, phi_ref, longAscNodes,
                      eccentricity, meanPerAna,
                      deltaF, f_min, f_max, f_ref,
                      lal.CreateDict(), approximant)
    If a list/tuple is returned, takes the first element (h_plus).
    """
    m1_SI = m1 * lal.MSUN_SI
    m2_SI = m2 * lal.MSUN_SI

    freqs_in = np.arange(f_min, f_max + df, df)
    dist_SI  = distance_mpc * 1e6 * lal.PC_SI

    spin1x = spin1y = spin1z = 0.0
    spin2x = spin2y = spin2z = 0.0
    eccentricity = 0.0
    meanPerAna   = 0.0
    incl         = 0.0
    phiRef       = 0.0
    longAscNodes = 0.0
    fRef = freqs_in[0]

    Hf_raw = lalsim.SimInspiralFD(
        m1_SI, m2_SI,
        spin1x, spin1y, spin1z,
        spin2x, spin2y, spin2z,
        dist_SI,
        incl, phiRef, longAscNodes,
        eccentricity, meanPerAna,
        df, f_min, f_max, fRef,
        lal.CreateDict(),
        approximant
    )

    if isinstance(Hf_raw, (list, tuple)):
        Hf_series = Hf_raw[0]
    else:
        Hf_series = Hf_raw

    length = Hf_series.data.length
    H0 = np.zeros(length, dtype=np.complex128)
    for idx in range(length):
        cplx = Hf_series.data.data[idx]
        H0[idx] = complex(cplx.real, cplx.imag)

    return freqs_in, H0


def interpolate_H0(freqs_full, freqs_fd, H0_fd):
    """
    Given:
      – freqs_full: full numpy array of positive freqs (one‐sided, e.g. rfftfreq(N,dt))
      – freqs_fd:   array from generate_fd_waveform (coarser grid)
      – H0_fd:      complex array on freqs_fd
    Returns:
      – H0_full: complex array of length freqs_full.shape, where
           H0_full[0] = 0.0 + 0.0j (DC)
           H0_full[1:] = interpolated complex waveform on freqs_full[1:].
    """
    H0_full = np.zeros_like(freqs_full, dtype=np.complex128)

    # Interpolate real and imag parts separately (so phase is preserved):
    real_interp = np.interp(freqs_full[1:], freqs_fd, np.real(H0_fd[1:]), left=0.0, right=0.0)
    imag_interp = np.interp(freqs_full[1:], freqs_fd, np.imag(H0_fd[1:]), left=0.0, right=0.0)

    H0_full[1:] = real_interp + 1j * imag_interp
    H0_full[0] = 0.0 + 0.0j

    return H0_full


# ──────────────────────────────────────────────────────────────────────────────
# 3) CORRECTION CALCULATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_corrections(H0, PSD2, phi_mp, freqs):
    """
    Compute both first‐order and second‐order analytic timing/phase corrections:

    First-order:
       Δt1  = [ ∫ f * (|H0|^2 / PSD2) * phi_mp  df ] / [ 2π * ∫ f^2 * (|H0|^2 / PSD2)  df ]
       Δφ1  = [ ∫ (|H0|^2 / PSD2) * phi_mp  df ] / [ ∫ (|H0|^2 / PSD2)  df ]

    Second-order (dropping mixed Δt1 terms, only leading phi_mp^2):
       Δt2  ≈ − [ ∫ f^2 * (|H0|^2 / PSD2) * phi_mp^2  df ] / [ (2π)^2 * ∫ f^2 * (|H0|^2 / PSD2)  df ]
       Δφ2  ≈ − [ ∫ (|H0|^2 / PSD2) * phi_mp^2  df ] / [ 2 * ∫ (|H0|^2 / PSD2)  df ]

    Inputs:
      – H0[0..M-1]    : complex FD waveform on freqs[0..M-1]
      – PSD2[0..M-1]  : one‐sided PSD used for data whitening
      – phi_mp[0..M-1]: minimum-phase whitening phase from PSD2
      – freqs[0..M-1] : positive‐frequency axis (Hz)
    Returns:
      – (dt1, dphi1, dt2, dphi2)
    """
    Wf = (np.abs(H0) ** 2) / PSD2

    fpos    = freqs[1:]
    Wfpos   = Wf[1:]
    Phi_pos = phi_mp[1:]

    num_t1 = np.trapz(fpos * Wfpos * Phi_pos, x=fpos)
    den_t1 = 2.0 * np.pi * np.trapz((fpos ** 2) * Wfpos, x=fpos)
    dt1    = num_t1 / den_t1

    num_phi1 = np.trapz(Wfpos * Phi_pos, x=fpos)
    den_phi1 = np.trapz(Wfpos, x=fpos)
    dphi1    = num_phi1 / den_phi1

    num_t2 = np.trapz((fpos ** 2) * Wfpos * (Phi_pos ** 2), x=fpos)
    den_t2 = (2.0 * np.pi) ** 2 * np.trapz((fpos ** 2) * Wfpos, x=fpos)
    dt2    = -num_t2 / den_t2

    num_phi2 = np.trapz(Wfpos * (Phi_pos ** 2), x=fpos)
    den_phi2 = 2.0 * np.trapz(Wfpos, x=fpos)
    dphi2    = -num_phi2 / den_phi2

    return dt1, dphi1, dt2, dphi2


# ──────────────────────────────────────────────────────────────────────────────
# 4) MATCHED‐FILTER UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def match_filter_time_series(h1_t, x2_t, dt):
    """
    Compute the **complex** matched‐filter time series Z(t_c) = ∫ h1*(τ) x2(τ + t_c) dτ
    via FFT.  Returns:
      • tc (length‐N array of lags in seconds, from -T/2 … +T/2−dt)
      • Z_t (length‐N complex array)
    """
    N = len(h1_t)

    H1_full = np.fft.fft(h1_t, n=N)
    X2_full = np.fft.fft(x2_t, n=N)

    corr_full = np.conj(H1_full) * X2_full
    z_unnorm  = np.fft.ifft(corr_full, n=N)
    z_t       = np.fft.fftshift(z_unnorm)

    idx = np.arange(N)
    tc  = (idx - (N // 2)) * dt

    return tc, z_t


# ──────────────────────────────────────────────────────────────────────────────
# 5) INJECTION / TIME‐DOMAIN PREPARATION
# ──────────────────────────────────────────────────────────────────────────────

def make_injection_time_series(H0, W2, freqs, N, dt, t_true, phi_true):
    """
    Build “centered” whitened‐data time series x2_centered(t), as complex.

    Steps:
      1) Construct the FD injection: H_sig(f) = H0(f)*exp(i φ_true)*exp(-2π i f t_true).
      2) Whiten in FD:          X2_pos(f)   = H_sig(f)*W2(f)  (one-sided).
      3) Mirror X2_pos → X2_full (length N) with Hermitian symmetry.
      4) IFFT → x2_t (complex length-N time series).
    """
    M = len(freqs)  # = N//2 + 1

    # 1) frequency‐domain phase shift for coalescence at t_true
    phase_shift = np.exp(-2j * np.pi * freqs * t_true)
    Hsig_f = H0 * np.exp(1j * phi_true) * phase_shift

    # 2) whiten by PSD2’s minimum‐phase filter
    X2_pos = Hsig_f * W2

    # 3) build the full length-N spectrum
    X2_full = np.zeros(N, dtype=complex)
    X2_full[:M] = X2_pos
    X2_full[M:] = np.conj(X2_pos[-2:0:-1])

    # 4) IFFT → full complex time series
    x2_t = np.fft.ifft(X2_full)

    return x2_t


# ──────────────────────────────────────────────────────────────────────────────
# 6) RESIDUAL‐WRAPPING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def wrap_time_residuals(t_hat_arr, t_true_arr, duration):
    """
    Given arrays of raw or corrected estimates t_hat and true times t_true (both length N),
    return a new array of residuals (t_hat - t_true) wrapped into [-duration/2, +duration/2).
    """
    Δ = t_hat_arr - t_true_arr
    res = (Δ + 1.5 * duration) % duration - 0.5 * duration
    return res


def wrap_phase_residuals(phi_hat_arr, phi_true_arr):
    """
    Given arrays of raw or corrected φ_hat and true φ_true (length N),
    return angle[ exp(i φ_hat) / exp(i φ_true) ], which is automatically in [-π, +π).
    """
    return np.angle(np.exp(1j * phi_hat_arr) / np.exp(1j * phi_true_arr))


# ──────────────────────────────────────────────────────────────────────────────
# 7) PLOTTING HISTOGRAMS
# ──────────────────────────────────────────────────────────────────────────────

def plot_residuals_histograms(
        raw_time_res, corr1_time_res, corr12_time_res,
        raw_phase_res, corr1_phase_res, corr12_phase_res
):
    """
    Given six arrays of residuals (timing and phase), plot two side‐by‐side histograms:
      – Left: timing residuals  [seconds]
      – Right: phase residuals   [radians]
    Each panel shows three overlaid histograms: raw / 1st-order / 1+2nd-order.

    The x‐limits are chosen “dynamically” to be +/- 10% beyond the largest absolute value
    among all the data in each panel.
    """
    all_t   = np.concatenate([raw_time_res, corr1_time_res, corr12_time_res])
    tmax    = np.max(np.abs(all_t)) if all_t.size > 0 else 1e-4
    t_lim   = tmax * 1.1

    all_phi = np.concatenate([raw_phase_res, corr1_phase_res, corr12_phase_res])
    ph_max  = np.max(np.abs(all_phi)) if all_phi.size > 0 else 0.1
    phi_lim = ph_max * 1.1

    fig = plt.figure(figsize=(12, 5))

    # Timing subplot
    ax1 = fig.add_subplot(1, 2, 1)
    bins_time = np.linspace(-t_lim, +t_lim, 100)
    ax1.hist(raw_time_res,    bins=bins_time, color='red',     alpha=0.6, label='Raw')
    ax1.hist(corr1_time_res,  bins=bins_time, color='blue',    alpha=0.6, label='1st-order')
    ax1.hist(corr12_time_res, bins=bins_time, color='magenta', alpha=0.6, label='1+2nd-order')
    ax1.axvline(0.0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel("Timing residual (t_hat - t_true) [s]")
    ax1.set_ylabel("Number of injections")
    ax1.set_title("Timing residuals: raw vs corrected")
    ax1.legend()
    ax1.set_xlim(-t_lim, +t_lim)

    # Phase subplot
    ax2 = fig.add_subplot(1, 2, 2)
    bins_phase = np.linspace(-phi_lim, +phi_lim, 100)
    ax2.hist(raw_phase_res,    bins=bins_phase, color='orange', alpha=0.6, label='Raw')
    ax2.hist(corr1_phase_res,  bins=bins_phase, color='green',  alpha=0.6, label='1st-order')
    ax2.hist(corr12_phase_res, bins=bins_phase, color='purple', alpha=0.6, label='1+2nd-order')
    ax2.axvline(0.0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel("Phase residual (phi_hat - phi_true) [rad]")
    ax2.set_ylabel("Number of injections")
    ax2.set_title("Phase residuals: raw vs corrected")
    ax2.legend()
    ax2.set_xlim(-phi_lim, +phi_lim)

    plt.tight_layout()
    plt.show()


def plot_residuals_scatter(
    t_true_arr,
    t_hat_raw_arr,
    t_hat_corr1_arr,
    t_hat_corr12_arr,
    phi_true_arr,
    phi_hat_raw_arr,
    phi_hat_corr1_arr,
    phi_hat_corr12_arr,
    duration,
):
    """
    Draw side‐by‐side scatterplots:
      – Left:  t_hat vs. t_true  for Raw, 1st‐order, and 1+2nd‐order corrections.
      – Right: phi_hat vs. phi_true for Raw, 1st‐order, and 1+2nd‐order corrections.
    """
    fig = plt.figure(figsize=(12, 5))

    # LEFT: timing
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(t_true_arr, t_hat_raw_arr,    color="red",     alpha=0.5, label="Raw")
    ax1.scatter(t_true_arr, t_hat_corr1_arr,  color="blue",    alpha=0.5, label="1st-order")
    ax1.scatter(t_true_arr, t_hat_corr12_arr, color="magenta", alpha=0.5, label="1+2nd-order")
    ax1.plot([0, duration], [0, duration], "k--", label="Ideal")
    ax1.set_xlabel("True $t_c$ [s]")
    ax1.set_ylabel("Estimated $t_c$ [s]")
    ax1.set_title("Coalescence Time: Raw vs. Corrections")
    ax1.set_xlim(0, duration)
    ax1.set_ylim(0, duration)
    ax1.legend(loc="upper left")

    # RIGHT: phase
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(phi_true_arr, phi_hat_raw_arr,    color="orange", alpha=0.5, label="Raw")
    ax2.scatter(phi_true_arr, phi_hat_corr1_arr,  color="green",  alpha=0.5, label="1st-order")
    ax2.scatter(phi_true_arr, phi_hat_corr12_arr, color="purple", alpha=0.5, label="1+2nd-order")
    ax2.plot([0, 2 * np.pi], [0, 2 * np.pi], "k--", label="Ideal")
    ax2.set_xlabel("True $\\phi_c$ [rad]")
    ax2.set_ylabel("Estimated $\\phi_c$ [rad]")
    ax2.set_title("Coalescence Phase: Raw vs. Corrections")
    ax2.set_xlim(0, 2 * np.pi)
    ax2.set_ylim(0, 2 * np.pi)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
