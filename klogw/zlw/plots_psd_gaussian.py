import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
Fs      = 4096.0        # Sampling frequency [Hz]
f0      = 150.0         # Center frequency of Gaussian bump [Hz]
sigma_f = 30.0          # Width (std) of Gaussian bump [Hz]
alpha   = 0.3           # Amplitude ratio of bump

# DFT grid
N         = 4096
omega     = np.linspace(0, np.pi, N)               # digital freq [0, π]
f_uniform = omega * Fs / (2*np.pi)                 # mapping to [Hz]

# --- Gaussian‐bump PSD & Cepstrum → W(ω) → w[n] ---
G   = np.exp(-0.5 * ((f_uniform - f0)/sigma_f)**2)
S   = 1.0 + alpha * G
L   = -0.5 * np.log(S)
c   = np.fft.ifft(L)
c_min = np.zeros_like(c)
c_min[0]      = c[0]
c_min[1:N//2] = 2*c[1:N//2]
c_min[N//2]   = c[N//2]
F   = np.fft.fft(c_min)
W   = np.exp(F)
w   = np.fft.ifft(W)

# --- Plotting ---
fig, (ax0, ax1, ax2) = plt.subplots(3,1,figsize=(8,10))

# (a) PSD & amplitude
ax_psd = ax0
ax_amp = ax_psd.twinx()
ax_psd.loglog(f_uniform, S,   color='C0', label='PSD $S(f)$')
ax_amp.loglog(f_uniform, np.abs(W), color='C1', label='|W(e^{iω})|')
ax_psd.set_title('Gaussian‐bump PSD & Whitening Amplitude')
ax_psd.set_xlabel('Frequency [Hz]')
ax_psd.set_ylabel('PSD', color='C0');  ax_amp.set_ylabel('Amplitude', color='C1')
ax_psd.grid(which='both', ls='--', alpha=0.5)
h0, l0 = ax_psd.get_legend_handles_labels()
h1, l1 = ax_amp.get_legend_handles_labels()
ax0.legend(h0+h1, l0+l1, loc='upper left')

# (b) phase
ax1.semilogx(f_uniform, np.angle(W), color='C1')
ax1.set_title('Whitening Filter Phase Response')
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel(r'Phase $\varphi(\omega)$ [rad]')
ax1.grid(which='both', ls='--', alpha=0.5)

# (c) impulse response (first 200 samples)
n    = np.arange(len(w))
wn   = w[:200]
mask = np.abs(wn) > 0    # drop the exact zeros

ax2.plot(n[:200][mask], wn.real[mask], color='C2', label='$\Re\{w[n]\}$')
ax2.plot(n[:200][mask], wn.imag[mask], color='C3', label='$\Im\{w[n]\}$')
# plain log: only plot the non‐zero points
ax2.plot(n[:200][mask], np.abs(wn)[mask], color='C4', linestyle='--', label='$|w[n]|$')
# ax2.set_yscale('log')
# OR, to use a symmetric log scale (uncomment below two lines)
ax2.set_yscale('symlog', linthresh=1e-6)
ax2.set_ylim(-1, 1)

ax2.set_title('Impulse Response (first 100 samples)')
ax2.set_xlabel('n')
ax2.set_xlim(-5, 100)
ax2.set_ylabel('Amplitude (SymLog)')
ax2.legend(loc='upper right')
ax2.grid(which='both', ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig('gaussian_bump_whitening_response_clean.pdf')
plt.show()
