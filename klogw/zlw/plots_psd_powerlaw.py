import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
Fs = 4096.0         # sampling rate [Hz]
p  = 2.0            # power-law exponent
N  = 4096           # FFT size (even)

# digital angular frequencies ω ∈ [0, 2π)
omega = np.linspace(0, 2*np.pi, N, endpoint=False)
# map to physical frequency f = (Fs/2π) ω ∈ [0, Fs)
f = omega * (Fs / (2*np.pi))

# avoid f=0 (amplitude would diverge), enforce DC=0
f[0] = f[1]
# ─────────────────────────────────────────────────────────────────────────────
# ANALYTIC PSD, AMPLITUDE, PHASE
# ─────────────────────────────────────────────────────────────────────────────
S = f**(-p)                            # PSD ~ f^{-p}
amp = S**(-0.5)                        # |W| = 1/sqrt(S) = f^{p/2}
phi = (p * np.pi / 4) * np.ones_like(f)# φ(ω) = (pπ/4) constant

# build the analytic complex frequency response
W_analytic = amp * np.exp(1j * phi)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD MINIMUM-PHASE IMPULSE RESPONSE via FOLDED CEPSTRUM
# ─────────────────────────────────────────────────────────────────────────────
# 1) form log-spectrum samples
F = np.log(W_analytic)                # = L + i φ
L = np.real(F)                         # should equal 0.5 * p * ln f
# 2) compute real cepstrum
c = np.fft.ifft(L)
# 3) fold into minimum-phase cepstrum
c_min = np.zeros_like(c)
c_min[0]      = c[0]
c_min[1:N//2] = 2*c[1:N//2]
c_min[N//2]   = c[N//2]
# 4) recover the *complex* log-spectrum
F_min = np.fft.fft(c_min)
# 5) exponentiate → min-phase freq response
W_min = np.exp(F_min)
# 6) impulse response
w = np.fft.ifft(W_min)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax0, ax1, ax2) = plt.subplots(3,1,figsize=(8,10))

# Top: PSD (blue) & amplitude (orange) on twin‐y
ax_psd = ax0
ax_amp = ax_psd.twinx()
ax_psd.loglog(f,   S,   color='C0', label='Power‐law PSD $f^{-p}$')
ax_amp.loglog(f,   amp, color='C1', label=r'$|W(e^{i\omega})|$')

ax_psd.set_ylabel(r'$S(f)$',    color='C0')
ax_amp.set_ylabel('Amplitude',  color='C1')
ax_psd.set_title(f'Power‐Law PSD (p={p}) and Whitening Amplitude')
ax_psd.set_xlabel('Frequency [Hz]')
ax_psd.grid(which='both', ls='--', alpha=0.5)

# combine legends
h0,l0 = ax_psd.get_legend_handles_labels()
h1,l1 = ax_amp.get_legend_handles_labels()
ax_psd.legend(h0+h1, l0+l1, loc='upper left')

# Middle: analytic phase
ax1.semilogx(f, phi, color='C2')
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel(r'Phase $\varphi(\omega)$ [rad]')
ax1.set_title('Whitening Filter Phase Response (analytic)')
ax1.grid(which='both', ls='--', alpha=0.5)

# Bottom: impulse response components (first 100 samples)
n = np.arange(100)
ax2.plot(n, w.real[:100], color='C3', label=r'$\Re\{w[n]\}$')
ax2.plot(n, w.imag[:100], color='C4', label=r'$\Im\{w[n]\}$')
ax2.plot(n, np.abs(w[:100]), color='C5', linestyle='--', label=r'$|w[n]|$')
ax2.set_xlabel('Sample index $n$')
ax2.set_ylabel('Amplitude')
ax2.set_title('Whitening Filter Impulse Response (minimum‐phase)')
ax2.set_yscale('symlog')
ax2.set_xlim(0,100)
ax2.set_ylim(-10000, 10000)
ax2.legend(loc='upper right')
ax2.grid(which='both', ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig('powerlaw_whitening_combined.pdf')
plt.show()
