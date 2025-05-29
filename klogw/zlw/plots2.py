#!/usr/bin/env python3
"""
Updated Python script that ensures axis labels are not cut off in PDF outputs.
We use plt.tight_layout() and bbox_inches='tight' to preserve labels and titles.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_compound_example_mag():
    w0 = 7.0
    zeta = 0.1
    w1 = 20.0

    # Numerator: w1*(s^2 + w0^2)
    num_an = [w1, 0, w1*(w0**2)]
    # Denominator: (s + w1)( s^2 + 2 zeta w0 s + w0^2 )
    den_an = [1, 2*zeta*w0 + w1, (w0**2 + 2*zeta*w0*w1), (w0**2)*w1]

    w_an = np.linspace(0, 40, 1000)
    w_resp, H_an = signal.freqs(num_an, den_an, worN=w_an)
    mag_an = np.abs(H_an)

    plt.figure(figsize=(5,3))
    plt.plot(w_resp, mag_an, 'b-')
    plt.title("compound_example_mag (analog) - Magnitude")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel(r"$|H(j\omega)|$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("compound_example_mag.pdf", bbox_inches='tight')
    plt.close()

def plot_notch_lowpass_mag():
    wc = 20.0
    w0 = 7.0
    zeta = 0.1
    # H_lp(s) = wc / (s+wc)
    num_lp = [wc]
    den_lp = [1, wc]
    # H_notch(s) = (s^2 + w0^2)/(s^2 + 2 zeta w0 s + w0^2)
    num_nt = [1, 0, w0**2]
    den_nt = [1, 2*zeta*w0, w0**2]

    num_c = np.convolve(num_lp, num_nt)
    den_c = np.convolve(den_lp, den_nt)

    w_an = np.linspace(0, 40, 1000)
    w_resp, H_an = signal.freqs(num_c, den_c, worN=w_an)
    mag_an = np.abs(H_an)

    plt.figure(figsize=(5,3))
    plt.plot(w_resp, mag_an, 'b-')
    plt.title("notch_lowpass_mag")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel(r"$|H(j\omega)|$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("notch_lowpass_mag.pdf", bbox_inches='tight')
    plt.close()

def plot_cascade_lowpass_mag():
    wc = 10.0
    # H2(s)= [wc/(s+wc)]^2 => num= wc^2, den= (s+wc)^2 => [1,2wc,wc^2]
    num = [wc**2]
    den = [1, 2*wc, wc**2]

    w_an = np.linspace(0, 100, 1000)
    w_resp, H_an = signal.freqs(num, den, worN=w_an)
    mag_an = np.abs(H_an)

    plt.figure(figsize=(5,3))
    plt.plot(w_resp, mag_an, 'b-')
    plt.title("cascade_lowpass_mag")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel(r"$|H(j\omega)|$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cascade_lowpass_mag.pdf", bbox_inches='tight')
    plt.close()

def plot_resonator_phase():
    w0 = 5.0
    zeta = 0.05
    num_an = [w0**2]
    den_an = [1, 2*zeta*w0, w0**2]

    w_an = np.linspace(0, 10, 1000)
    w_resp, H_an = signal.freqs(num_an, den_an, worN=w_an)
    phase_an = np.unwrap(np.angle(H_an)) * 180/np.pi

    plt.figure(figsize=(5,3))
    plt.plot(w_resp, phase_an, 'r-')
    plt.title("resonator_phase")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel("Phase (degrees)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("resonator_phase.pdf", bbox_inches='tight')
    plt.close()

def plot_resonator_mag():
    w0 = 5.0
    zeta = 0.05
    num_an = [w0**2]
    den_an = [1, 2*zeta*w0, w0**2]

    w_an = np.linspace(0, 10, 1000)
    w_resp, H_an = signal.freqs(num_an, den_an, worN=w_an)
    mag_an = np.abs(H_an)

    plt.figure(figsize=(5,3))
    plt.plot(w_resp, mag_an, 'b-')
    plt.title("resonator_mag")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel(r"$|H(j\omega)|$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("resonator_mag.pdf", bbox_inches='tight')
    plt.close()

def plot_notch_phase():
    w0 = 15.0
    zeta = 0.1
    num_an = [1, 0, w0**2]
    den_an = [1, 2*zeta*w0, w0**2]

    w_an = np.linspace(0, 30, 1000)
    w_resp, H_an = signal.freqs(num_an, den_an, worN=w_an)
    phase_an = np.unwrap(np.angle(H_an))*180/np.pi

    plt.figure(figsize=(5,3))
    plt.plot(w_resp, phase_an, 'r-')
    plt.title("notch_phase")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel("Phase (deg)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("notch_phase.pdf", bbox_inches='tight')
    plt.close()

def plot_notch_mag():
    w0 = 15.0
    zeta = 0.1
    num_an = [1, 0, w0**2]
    den_an = [1, 2*zeta*w0, w0**2]

    w_an = np.linspace(0, 30, 1000)
    w_resp, H_an = signal.freqs(num_an, den_an, worN=w_an)
    mag_an = np.abs(H_an)

    plt.figure(figsize=(5,3))
    plt.plot(w_resp, mag_an, 'b-')
    plt.title("notch_mag")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel(r"$|H(j\omega)|$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("notch_mag.pdf", bbox_inches='tight')
    plt.close()

def plot_bandpass_phase():
    w0 = 5.0
    zeta = 0.3
    num_an = [1, 0]  # s
    den_an = [1, 2*zeta*w0, w0**2]

    w_an = np.linspace(0, 15, 1000)
    w_resp, H_an = signal.freqs(num_an, den_an, worN=w_an)
    phase_an = np.unwrap(np.angle(H_an))*180/np.pi

    plt.figure(figsize=(5,3))
    plt.plot(w_resp, phase_an, 'r-')
    plt.title("bandpass_phase")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel("Phase (deg)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bandpass_phase.pdf", bbox_inches='tight')
    plt.close()

def plot_bandpass_mag():
    w0 = 5.0
    zeta = 0.3
    num_an = [1, 0]
    den_an = [1, 2*zeta*w0, w0**2]

    w_an = np.linspace(0, 15, 1000)
    w_resp, H_an = signal.freqs(num_an, den_an, worN=w_an)
    mag_an = np.abs(H_an)

    plt.figure(figsize=(5,3))
    plt.plot(w_resp, mag_an, 'b-')
    plt.title("bandpass_mag")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel(r"$|H(j\omega)|$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bandpass_mag.pdf", bbox_inches='tight')
    plt.close()

def plot_highpass_mag():
    """
    Generate highpass_mag.pdf:
      Plots only the magnitude response |H_HP(jω)| for an analog first-order high-pass.
    """
    # Transfer function: H_HP(s)= s/(s+1)
    # numerator s => [1, 0], denominator => [1, 1]
    num_hp = [1, 0]
    den_hp = [1, 1]

    # Frequency range of interest:
    w_an = np.linspace(0, 10, 1000)
    w_resp, H_an = signal.freqs(num_hp, den_hp, worN=w_an)
    mag_an = np.abs(H_an)

    plt.figure(figsize=(5,3))
    plt.plot(w_resp, mag_an, 'b-')
    plt.title("highpass_mag")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel(r"$|H(j\omega)|$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("highpass_mag.pdf", bbox_inches='tight')
    plt.close()

def plot_highpass_phase():
    """
    Generate highpass_phase.pdf:
      Plots only the phase response arg(H_HP(jω)) for an analog first-order high-pass.
    """
    num_hp = [1, 0]
    den_hp = [1, 1]

    w_an = np.linspace(0, 10, 1000)
    w_resp, H_an = signal.freqs(num_hp, den_hp, worN=w_an)
    phase_an = np.unwrap(np.angle(H_an))*180.0/np.pi

    plt.figure(figsize=(5,3))
    plt.plot(w_resp, phase_an, 'r-')
    plt.title("highpass_phase")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel("Phase (deg)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("highpass_phase.pdf", bbox_inches='tight')
    plt.close()

def main():
    plot_compound_example_mag()
    plot_notch_lowpass_mag()
    plot_cascade_lowpass_mag()
    plot_resonator_phase()
    plot_resonator_mag()
    plot_notch_phase()
    plot_notch_mag()
    plot_bandpass_phase()
    plot_bandpass_mag()
    plot_highpass_mag()
    plot_highpass_phase()
    print("All missing figures generated successfully (with improved layout).")

if __name__=="__main__":
    main()
