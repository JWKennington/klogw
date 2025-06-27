#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_whitening_examples.py

Generate test‐case PSDs and their minimum‐phase whitening amplitude responses,
and save each as a PDF.
Dependencies:
  • numpy
  • matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
Fs = 4096.0            # [Hz]
Nf = 2000              # number of frequency points
f = np.linspace(1e-2, Fs/2, Nf)  # avoid f=0

# =============================================================================
# Example 1: Lorentzian PSD (single‐pole roll‐off)
#    S(f) = 1 / [1 + (f/fc)^2],    |W| = sqrt(1 + (f/fc)^2)
# =============================================================================
fc = 100.0  # corner freq [Hz]
S1 = 1.0 / (1.0 + (f/fc)**2)
W1 = np.sqrt(1.0 / S1)

plt.figure(figsize=(6,4))
plt.loglog(f, S1,   label='PSD (Lorentzian)')
plt.loglog(f, W1,   label='Whitening |W|')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.title('Lorentzian PSD & Whitening Amplitude')
plt.legend()
plt.grid(which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('lorentzian_whitening.pdf')

# =============================================================================
# Example 2: Exponential PSD (high‐freq decay)
#    S(f) = exp(-f/f0),    |W| = exp(+f/(2 f0))
# =============================================================================
f0 = 300.0  # decay constant [Hz]
S2 = np.exp(-f / f0)
W2 = np.exp( +f / (2.0 * f0) )

plt.figure(figsize=(6,4))
plt.semilogy(f, S2,   label='PSD (Exponential)')
plt.semilogy(f, W2,   label='Whitening |W|')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.title('Exponential PSD & Whitening Amplitude')
plt.legend()
plt.grid(which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('exponential_whitening.pdf')

# =============================================================================
# Example 3: Gaussian‐bump PSD (localized resonance)
#    S(f) = exp[-(f-f0)^2/(2σ^2)],    |W| = exp[+(f-f0)^2/(4σ^2)]
# =============================================================================
f0_bump = 200.0  # center freq [Hz]
sigma = 30.0     # bandwidth [Hz]
S3 = np.exp(-0.5 * ((f - f0_bump)/sigma)**2)
W3 = np.exp( +0.25 * ((f - f0_bump)/sigma)**2 )

plt.figure(figsize=(6,4))
plt.plot(f, S3,   label='PSD (Gaussian bump)')
plt.plot(f, W3,   label='Whitening |W|')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.title('Gaussian‐Bump PSD & Whitening Amplitude')
plt.legend()
plt.grid(ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('gaussian_bump_whitening.pdf')

# =============================================================================
# Example 4: Power‐law PSD S(f) ∝ f^{-α}
#    S(f) = (f/fref)^{-α},    |W| = (f/fref)^{+α/2}
# =============================================================================
alpha = 2.0
fref  = 100.0  # reference freq [Hz]
S4 = (f / fref)**(-alpha)
W4 = (f / fref)**(+alpha/2)

plt.figure(figsize=(6,4))
plt.loglog(f, S4,   label=r'PSD ($f^{-%.1f}$)' % alpha)
plt.loglog(f, W4,   label=r'Whitening $|W|\propto f^{%.1f}$' % (alpha/2))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.title('Power‐Law PSD & Whitening Amplitude')
plt.legend()
plt.grid(which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('powerlaw_whitening.pdf')

print("All figures saved as PDF.")
