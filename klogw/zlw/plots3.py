#!/usr/bin/env python3
"""
Example script demonstrating LALSuite's SimInspiralChooseTDWaveform with correct argument order,
plus simple band-pass + notch filters, a toy 'whitening', and a naive matched filter approach.
Saves PDF plots.  This is strictly for demonstration.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lal
import lalsimulation

from scipy.signal import butter, iirnotch, filtfilt
from numpy.fft import rfft, irfft

def main():
    # -----------------------------------------------------
    # 1) CBC waveform generation: 30+30 Msun at 400Mpc, f_min=20Hz, IMRPhenomD
    #    using correct argument order for SimInspiralChooseTDWaveform
    # -----------------------------------------------------
    m1_kg      = 30.0 * lal.MSUN_SI
    m2_kg      = 30.0 * lal.MSUN_SI
    s1x = s1y = s1z = 0.0
    s2x = s2y = s2z = 0.0
    distance_m = 400e6 * lal.PC_SI  # 400 Mpc
    inclination= 0.0
    phiRef     = 0.0
    longAscNodes = 0.0
    eccentricity = 0.0
    meanPerAno   = 0.0
    deltaT    = 1.0/4096.0
    f_min     = 20.0
    f_ref     = f_min
    params    = lal.CreateDict()  # empty dict
    approximant = lalsimulation.IMRPhenomD

    # The actual waveforms (time-domain):
    hplus, hcross = lalsimulation.SimInspiralChooseTDWaveform(
        m1_kg,
        m2_kg,
        s1x, s1y, s1z,
        s2x, s2y, s2z,
        distance_m,
        inclination,
        phiRef,
        longAscNodes,
        eccentricity,
        meanPerAno,
        deltaT,
        f_min,
        f_ref,
        params,
        approximant
    )

    hp_arr = hplus.data.data
    Nsig   = hplus.data.length

    # normalize amplitude
    sig_max = np.max(np.abs(hp_arr))
    signal  = hp_arr / sig_max

    # -----------------------------------------------------
    # 2) Make a longer data array, embed signal, add noise
    # -----------------------------------------------------
    data_duration = 8.0
    fs = 1.0 / deltaT
    Nsamps = int(data_duration * fs)
    data = np.zeros(Nsamps, dtype=float)

    start_idx = Nsamps//2 - Nsig//2
    data[start_idx : start_idx+Nsig] += signal

    np.random.seed(42)
    noise = 0.3 * np.random.normal(0,1,Nsamps)
    data += noise

    # -----------------------------------------------------
    # 3) A naive toy "whiten" function
    # -----------------------------------------------------
    def simple_whiten(x):
        Xf = rfft(x)
        mag = np.abs(Xf)
        eps = 1e-12
        inv_mag = 1.0 / np.maximum(mag, eps)
        Xf_white = Xf * inv_mag
        return irfft(Xf_white, n=len(x))

    white_data = simple_whiten(data)

    # -----------------------------------------------------
    # 4) band-pass + notch, then whiten
    # -----------------------------------------------------
    def bandpass_notch(x, fsample):
        lowcut, highcut = 20.0, 300.0
        b, a = butter(4, [lowcut/(fsample/2), highcut/(fsample/2)], btype='band')
        y = filtfilt(b, a, x)
        for f0 in [60.0,120.0,180.0]:
            bn, an = iirnotch(f0/(fsample/2), Q=30)
            y = filtfilt(bn, an, y)
        return y

    bp_data = bandpass_notch(data, fs)
    white_bp_data = simple_whiten(bp_data)

    # -----------------------------------------------------
    # 5) matched_filter_toy: cross-correlation in freq domain
    # -----------------------------------------------------
    def matched_filter_toy(wdata, wtemp):
        L = len(wdata)
        if len(wtemp)<L:
            wtemp = np.pad(wtemp, (0,L-len(wtemp)))
        elif len(wtemp)>L:
            wtemp = wtemp[:L]
        Wd = rfft(wdata)
        Wt = rfft(wtemp)
        corr_f = Wd * np.conjugate(Wt)
        corr_t = irfft(corr_f, n=L)
        return corr_t

    # build a template array, same length, whiten it
    template = np.zeros(Nsamps, dtype=float)
    template[start_idx : start_idx+Nsig] = signal
    white_template = simple_whiten(template)

    snr_white = matched_filter_toy(white_data, white_template)
    snr_bp    = matched_filter_toy(white_bp_data, white_template)

    idx_peak_white = np.argmax(np.abs(snr_white))
    idx_peak_bp    = np.argmax(np.abs(snr_bp))
    peak_val_white = snr_white[idx_peak_white]
    peak_val_bp    = snr_bp[idx_peak_bp]
    print(f"Peak corr (whitened) = {peak_val_white:.3f}, index={idx_peak_white}")
    print(f"Peak corr (bp+whitened)= {peak_val_bp:.3f}, index={idx_peak_bp}")

    # -----------------------------------------------------
    # 6) plotting
    # -----------------------------------------------------
    times = np.arange(Nsamps)*deltaT
    true_tc_idx = start_idx + np.argmax(signal)
    true_tc = times[true_tc_idx]

    # waveforms
    plt.figure(figsize=(8,4))
    plt.plot(times, data, label='Raw data+signal', color='gray', alpha=0.5)
    plt.plot(times, white_data, label='Whitened', color='C0')
    plt.plot(times, white_bp_data, label='BP+notch+white', color='C1', linestyle='--')
    plt.axvline(true_tc, color='k', linestyle='-.', label='True $t_c$')
    plt.xlim(true_tc-0.3, true_tc+0.2)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Time-domain: raw vs. filtered (LALSuite waveform)")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("filtered_waveforms_example.pdf", bbox_inches='tight')
    plt.close()

    # matched filter output
    def windowaround(arr, peak, window=200):
        left = max(0, peak-window)
        right= min(len(arr), peak+window)
        return left, right

    left_w, right_w = windowaround(snr_white, idx_peak_white, 200)
    tw = times[left_w:right_w]

    plt.figure(figsize=(8,4))
    plt.plot(tw, snr_white[left_w:right_w], label='Corr(whitened)', color='C0')
    plt.plot(tw, snr_bp[left_w:right_w],    label='Corr(bp+white)', color='C1', linestyle='--')
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

    print("Saved: filtered_waveforms_example.pdf, snr_outputs_example.pdf")

if __name__=="__main__":
    main()
