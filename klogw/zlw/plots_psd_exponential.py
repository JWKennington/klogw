import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
Fs = 4096.0       # Sampling frequency [Hz]
f0 = 300.0        # Exponential PSD e-folding frequency [Hz]
beta = Fs / (2 * np.pi * f0)

# Frequency array for PSD & response plots (1 Hz → Nyquist)
f = np.logspace(np.log10(1), np.log10(Fs/2), 5000)
omega = 2 * np.pi * f / Fs  # Digital angular frequency [rad/sample]

# --- Exponential PSD and Whitening Filter ---
S_exp = np.exp(-f / f0)                      # PSD: S(f) = exp(-f/f0)
amplitude = np.exp(beta * omega / 2)         # |W(e^{iω})|
phase = 0.5 * beta * (omega - np.pi)         # φ(ω) = (β/2)(ω - π)

# --- Impulse Response Kernel ---
# Closed-form: w[n] = e^{-i βπ/2} * (β/2 + i n) / [ (β/2)^2 + n^2 ], n>=0
N_kernel = 512
n = np.arange(N_kernel)
wn = np.exp(-1j * beta * np.pi / 2) * (beta/2 + 1j * n) / ((beta/2)**2 + n**2)

# --- Plotting ---
fig, (ax_top, ax_phase, ax_time) = plt.subplots(3, 1, figsize=(8, 10))

# Top: PSD and Amplitude on twin axes
ax_psd = ax_top
ax_amp = ax_psd.twinx()
ax_psd.loglog(f, S_exp, color='C0', label='Exponential PSD')
ax_psd.set_ylabel(r'$S(f)$', color='C0')
ax_amp.loglog(f, amplitude, color='C1', label=r'$|W(e^{i\omega})|$')
ax_amp.set_ylabel('Amplitude', color='C1')
ax_psd.set_xlabel('Frequency [Hz]')
ax_psd.set_title(r'Exponential PSD and Whitening Amplitude ($f_0=300\,$Hz)')
ax_psd.grid(which='both', ls='--', alpha=0.5)
lines, labels = ax_psd.get_legend_handles_labels()
lines2, labels2 = ax_amp.get_legend_handles_labels()
ax_top.legend(lines+lines2, labels+labels2, loc='upper left')

# Middle: Phase response
ax_phase.semilogx(f, phase, color='C1')
ax_phase.set_xlabel('Frequency [Hz]')
ax_phase.set_ylabel(r'Phase $\varphi(\omega)$ [rad]')
ax_phase.set_title('Whitening Filter Phase Response')
ax_phase.grid(which='both', ls='--', alpha=0.5)

# Bottom: Impulse response magnitude
ax_time.plot(n, np.abs(wn), color='C2')
ax_time.set_xlabel('Sample index $n$')
ax_time.set_ylabel(r'$|w[n]|$')
ax_time.set_title('Whitening Filter Impulse Response Magnitude')
ax_time.set_yscale('log')
ax_time.grid(which='both', ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig('exponential_whitening_response.pdf')
plt.show()
