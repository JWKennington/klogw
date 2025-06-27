import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
Fs = 4096.0        # Sampling frequency [Hz]
fc = 300.0         # Lorentzian corner frequency [Hz]
omega_c = 2 * np.pi * fc / Fs  # Digital corner [rad/sample]

# Frequency array for PSD & response plots (1 Hz → Nyquist)
f = np.logspace(np.log10(1), np.log10(Fs/2), 5000)
omega = 2 * np.pi * f / Fs  # Digital angular frequency [rad/sample]

# --- Lorentzian PSD and Whitening Filter ---
S_lor = 1.0 / (1.0 + (f / fc)**2)       # S(f) = 1 / [1 + (f/fc)^2]
W = 1 + 1j * (omega / omega_c)         # W(e^{iω}) = 1 + i (ω/ω_c)
amplitude = np.abs(W)
phase     = np.angle(W)                # φ(ω) = arg[1 + i ω/ω_c]

# --- Impulse Response Kernel (corrected) ---
N_kernel = 512
n = np.arange(N_kernel)
wn = np.zeros(N_kernel, dtype=complex)
wn[0] = 1.0
wn[1:] = (-1)**(n[1:]+1) / (omega_c * n[1:])  # sign flips

# --- Plotting ---
fig, (ax_top, ax_phase, ax_kernel) = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

# Top: PSD and Amplitude on twin axes
ax_psd = ax_top
ax_amp = ax_psd.twinx()
ax_psd.loglog(f, S_lor, color='C0', linewidth=2, label='Lorentzian PSD')
ax_psd.set_ylabel(r'$S(f)$', color='C0')
ax_psd.tick_params(axis='y', labelcolor='C0')

ax_amp.loglog(f, amplitude, color='C1', linewidth=2, label=r'$|W(e^{i\omega})|$')
ax_amp.set_ylabel('Amplitude', color='C1')
ax_amp.tick_params(axis='y', labelcolor='C1')

ax_psd.set_xlabel('Frequency [Hz]')
ax_psd.set_title(r'Lorentzian PSD and Whitening Amplitude ($f_c = 300\,$Hz)')
ax_psd.grid(which='both', linestyle='--', alpha=0.5)

lines1, labels1 = ax_psd.get_legend_handles_labels()
lines2, labels2 = ax_amp.get_legend_handles_labels()
ax_top.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Middle: Phase response
ax_phase.semilogx(f, phase, color='C1', linewidth=2)
ax_phase.set_xlabel('Frequency [Hz]')
ax_phase.set_ylabel(r'Phase $\varphi(\omega)$ [rad]')
ax_phase.set_title('Whitening Filter Phase Response')
ax_phase.grid(which='both', linestyle='--', alpha=0.5)

# Bottom: Real, Imag, and Magnitude of w[n]
ax_kernel.plot(n, wn.real, color='C2',     label=r'$w[n]$')
ax_kernel.plot(n, np.abs(wn), color='C4', linestyle='--', label=r'$|w[n]|$')
ax_kernel.set_xlabel('Sample index $n$')
ax_kernel.set_ylabel('Amplitude')
ax_kernel.set_title('Whitening Filter Impulse Response Components')
# Set x limits to only show through sample 50
ax_kernel.set_xlim(0, 50)
ax_kernel.legend(loc='upper right')
ax_kernel.grid(which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('lorentzian_whitening_all_components.pdf')
plt.show()
