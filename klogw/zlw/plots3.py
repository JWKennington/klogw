#!/usr/bin/env python3
"""
Example script illustrating how to generate a CBC waveform using lalsimulation,
then apply band-pass, notch, and a toy 'whitening' approach, and finally do a
simple matched filter demonstration.  Outputs PDF plots.

Requirements:
 - lalsimulation, lal (from lalsuite)
 - numpy, scipy, matplotlib
 - (optional) a more realistic PSD for noise shaping

Note: This script is for demonstration only. Real LIGO analyses use more rigorous
methods for PSD estimation, noise simulation, and matched filtering.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # for headless usage; remove if you want interactive
import matplotlib.pyplot as plt

import lal
import lalsimulation

from scipy.signal import butter, iirnotch, filtfilt
from numpy.fft import rfft, irfft

def main():
    # -----------------------------------------------------
    # 1) Generate a CBC waveform with LALSuite (SimInspiralChooseTDWaveform)
    #    We'll do a 30+30 Msun BBH at 400 Mpc, f_lower=20 Hz, approx=IMRPhenomD
    #    The function signature typically is:
    #      SimInspiralChooseTDWaveform(phiRef, deltaT, m1, m2, S1x,S1y,S1z, S2x,S2y,S2z,
    #                                  fMin, fRef, distance, inclination, lambda1,lambda2,
    #                                  approximant)
    # -----------------------------------------------------

    # Parameters
    phiRef     = 0.0
    deltaT     = 1.0 / 4096.0
    m1_kg      = 30.0 * lal.MSUN_SI
    m2_kg      = 30.0 * lal.MSUN_SI
    S1x = S1y = S1z = 0.0
    S2x = S2y = S2z = 0.0
    fMin       = 20.0
    fRef       = fMin   # just the same, for simplicity
    distance_m = 400e6 * lal.PC_SI  # 400 Mpc
    inclination= 0.0
    lambda1 = 0.0
    lambda2 = 0.0

    approximant = lalsimulation.IMRPhenomD

    # Generate plus/cross
    hplus, hcross = lalsimulation.SimInspiralChooseTDWaveform(
        phiRef,
        deltaT,
        m1_kg, m2_kg,
        S1x, S1y, S1z,
        S2x, S2y, S2z,
        fMin, fRef,
        distance_m,
        inclination,
        lambda1, lambda2,
        approximant
    )
    # We'll just use hplus as our signal
    hp_arr = hplus.data.data
    Nsig   = hplus.data.length

    # Normalize amplitude to 1 for convenience
    sig_max = np.max(np.abs(hp_arr))
    signal  = hp_arr / sig_max

    # -----------------------------------------------------
    # 2) Place waveform in a longer data array & add noise
    # -----------------------------------------------------
    data_duration = 8.0  # total 8 seconds
    fs = 1.0 / deltaT
    Nsamps = int(data_duration * fs)
    data = np.zeros(Nsamps, dtype=float)

    start_idx = Nsamps//2 - Nsig//2
    data[start_idx : start_idx+Nsig] += signal

    # Add simple white noise for demonstration
    np.random.seed(42)
    noise = 0.3 * np.random.normal(0, 1, Nsamps)
    data += noise

    # Store a copy of the "pure signal" portion (for reference if needed)
    pure_signal = data[start_idx : start_idx+Nsig] - noise[start_idx : start_idx+Nsig]

    # -----------------------------------------------------
    # 3) A naive 'whiten' function in freq domain (toy)
    # -----------------------------------------------------
    def simple_whiten(x):
        Xf = rfft(x)
        mag = np.abs(Xf)
        # avoid zeros
        floor = 1e-12
        invmag = 1.0 / np.maximum(mag, floor)
        Xf_white = Xf * invmag
        return irfft(Xf_white, n=len(x))

    white_data = simple_whiten(data)

    # -----------------------------------------------------
    # 4) Band-pass + Notch, then whiten
    # -----------------------------------------------------
    def bandpass_and_notch(x, fsample):
        lowcut, highcut = 20.0, 300.0
        b, a = butter(4, [lowcut/(fsample/2), highcut/(fsample/2)], btype='band')
        y = filtfilt(b, a, x)
        # Notches
        for f0 in [60.0, 120.0, 180.0]:
            bn, an = iirnotch(f0/(fsample/2), Q=30)
            y = filtfilt(bn, an, y)
        return y

    bp_data = bandpass_and_notch(data, fs)
    white_bp_data = simple_whiten(bp_data)

    # -----------------------------------------------------
    # 5) Simple toy matched filter by cross-correlation in freq domain
    # -----------------------------------------------------
    def matched_filter_toy(wdata, wtemp):
        if len(wtemp) < len(wdata):
            wtemp = np.pad(wtemp, (0, len(wdata)-len(wtemp)), 'constant')
        elif len(wtemp)>len(wdata):
            wtemp = wtemp[:len(wdata)]
        Wd = rfft(wdata)
        Wt = rfft(wtemp)
        corr_f = Wd * np.conjugate(Wt)
        corr_t = irfft(corr_f, n=len(wdata))
        return corr_t

    # Make a 'template': same signal array placed in zero array, then whiten
    template = np.zeros(Nsamps)
    template[start_idx : start_idx+Nsig] = signal
    white_template = simple_whiten(template)

    snr_white = matched_filter_toy(white_data, white_template)
    snr_bp    = matched_filter_toy(white_bp_data, white_template)

    idx_peak_white = np.argmax(np.abs(snr_white))
    idx_peak_bp    = np.argmax(np.abs(snr_bp))
    peak_val_white = snr_white[idx_peak_white]
    peak_val_bp    = snr_bp[idx_peak_bp]

    print(f"Peak corr (whitened): {peak_val_white:.3f} at index {idx_peak_white}")
    print(f"Peak corr (bp+notch+white): {peak_val_bp:.3f} at index {idx_peak_bp}")

    # -----------------------------------------------------
    # 6) Plot time-domain waveforms
    # -----------------------------------------------------
    times = np.arange(Nsamps) * deltaT
    true_tc_idx = start_idx + np.argmax(signal)
    true_tc = times[true_tc_idx]

    plt.figure(figsize=(8,4))
    plt.plot(times, data, label='Raw data + signal', color='gray', alpha=0.5)
    plt.plot(times, white_data, label='Whitened', color='C0')
    plt.plot(times, white_bp_data, label='Band-passed+notched+whitened',
             color='C1', linestyle='--')
    plt.axvline(true_tc, color='k', linestyle='-.', label='True $t_c$')
    plt.xlim(true_tc - 0.3, true_tc + 0.2)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude (arb. units)")
    plt.title("Time-domain: raw vs. filtered (LALSuite waveforms)")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("filtered_waveforms_example.pdf", bbox_inches='tight')
    plt.close()

    # -----------------------------------------------------
    # 7) Plot the correlation outputs around the peak
    # -----------------------------------------------------
    window = 200
    left   = max(idx_peak_white - window, 0)
    right  = min(idx_peak_white + window, Nsamps)
    tw = times[left:right]

    plt.figure(figsize=(8,4))
    plt.plot(tw, snr_white[left:right], label='Corr (whitened)', color='C0')
    plt.plot(tw, snr_bp[left:right], label='Corr (bp+white)', color='C1', linestyle='--')
    plt.axvline(times[idx_peak_white], color='C0', linestyle=':')
    plt.axvline(times[idx_peak_bp], color='C1', linestyle=':')
    plt.axvline(true_tc, color='k', linestyle='-.', label='True $t_c$')
    plt.xlabel("Time [s]")
    plt.ylabel("Cross-correlation amplitude")
    plt.title("Toy matched filter output")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("snr_outputs_example.pdf", bbox_inches='tight')
    plt.close()

    print("Plots saved: filtered_waveforms_example.pdf, snr_outputs_example.pdf")

if __name__ == "__main__":
    main()
