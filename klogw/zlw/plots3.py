#!/usr/bin/env python3
"""
Improved script demonstrating:
  1) Generating a CBC waveform with LALSuite's SimInspiralChooseTDWaveform
  2) Adding noise, applying band-pass + notches
  3) Doing a more realistic whitening (via Welch PSD estimate)
  4) Toy matched filtering
  5) Plotting waveforms on separate subplots so each is visible

Produces:
  * filtered_waveforms_example.pdf
  * snr_outputs_example.pdf

This should avoid the 'tiny line near y=0' effect by scaling or separate subplots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lal
import lalsimulation

from scipy.signal import butter, iirnotch, filtfilt, welch
from numpy.fft import rfft, irfft

def main():
    # -----------------------------------------------------
    # 1) CBC waveform generation
    # -----------------------------------------------------
    m1_kg      = 30.0 * lal.MSUN_SI
    m2_kg      = 30.0 * lal.MSUN_SI
    s1x = s1y = s1z = 0.0
    s2x = s2y = s2z = 0.0
    distance_m = 400e6 * lal.PC_SI
    inclination= 0.0
    phiRef     = 0.0
    longAscNodes = 0.0
    eccentricity = 0.0
    meanPerAno   = 0.0
    deltaT    = 1.0/4096.0
    f_min     = 20.0
    f_ref     = f_min
    params    = lal.CreateDict()
    approximant = lalsimulation.IMRPhenomD

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
    sig_max = np.max(np.abs(hp_arr))
    signal = hp_arr / sig_max  # normalized

    # -----------------------------------------------------
    # 2) Build data array (8 s) + noise
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

    # store for reference if needed
    times = np.arange(Nsamps)*deltaT
    true_tc_idx = start_idx + np.argmax(signal)
    true_tc = times[true_tc_idx]

    # -----------------------------------------------------
    # 3) Band-pass + Notch filter
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

    # -----------------------------------------------------
    # 4) We'll estimate PSD via Welch, then do freq-domain weighting
    # -----------------------------------------------------
    # A small function to do 'realistic' whitening
    #  uses the PSD from welch in linear freq spacing, merges it
    def welch_whiten(x):
        # compute PSD with welch => (f,Pxx)
        seglen = 2**14  # or some power of 2
        overlap = seglen//2
        f_psd, p_psd = welch(x, fs, nperseg=seglen, noverlap=overlap)
        # p_psd is the PSD estimate at freq f_psd

        # next, build freq axis for rfft => freq spacing is fs/Nsamps
        freqs = np.fft.rfftfreq(len(x), d=1.0/fs)

        # we will do a simple interpolation => p_psd is ~ len(f_psd)
        # map onto freqs
        psd_interp = np.interp(freqs, f_psd, p_psd)
        # flatten zero => small floor
        eps=1e-20
        psd_interp = np.maximum(psd_interp, eps)

        Xf = rfft(x)
        # multiply by 1/sqrt(psd)
        Xf_white = Xf / np.sqrt(psd_interp)
        # keep phase as is => zero-phase approach or we can discard phase
        # but let's keep the real data => inverse rfft
        x_white = irfft(Xf_white, n=len(x))
        return x_white

    white_data = welch_whiten(data)
    white_bp_data = welch_whiten(bp_data)

    # also whiten a template
    template = np.zeros(Nsamps, dtype=float)
    template[start_idx:start_idx+Nsig] = signal
    white_template = welch_whiten(template)

    # -----------------------------------------------------
    # 5) matched filter by cross-correlation in freq domain
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

    snr_white = matched_filter_toy(white_data, white_template)
    snr_bp    = matched_filter_toy(white_bp_data, white_template)

    idx_peak_white = np.argmax(np.abs(snr_white))
    idx_peak_bp    = np.argmax(np.abs(snr_bp))
    peak_val_white = snr_white[idx_peak_white]
    peak_val_bp    = snr_bp[idx_peak_bp]
    print(f"Peak corr (whitened) = {peak_val_white:.3f}, index={idx_peak_white}")
    print(f"Peak corr (bp+whitened)= {peak_val_bp:.3f}, index={idx_peak_bp}")

    # -----------------------------------------------------
    # 6) Plot waveforms => separate subplots
    # -----------------------------------------------------
    fig, axes = plt.subplots(3,1, figsize=(8,8), sharex=True)

    axes[0].plot(times, data, label='Raw data+signal', color='gray', alpha=0.5)
    axes[0].axvline(true_tc, color='k', linestyle='-.', label='True $t_c$')
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Raw + noise (and embedded signal)")
    axes[0].legend()

    axes[1].plot(times, white_data, label='Whitened', color='C0')
    axes[1].plot(times, white_bp_data, label='BP+Notch+Whitened', color='C1', linestyle='--')
    axes[1].axvline(true_tc, color='k', linestyle='-.')
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Whitened vs. band-pass+notch+whitened")
    axes[1].legend()

    axes[2].plot(times, bp_data, label='Band-pass+notched data', color='C2')
    axes[2].axvline(true_tc, color='k', linestyle='-.')
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_title("Just band-passed + notched, pre-whitening")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("filtered_waveforms_example.pdf", bbox_inches='tight')
    plt.close()

    # -----------------------------------------------------
    # 7) Plot matched filter outputs in separate subplots
    # -----------------------------------------------------
    fig2, ax2 = plt.subplots(2,1, figsize=(8,6), sharex=True)

    def windowaround(arr, peak, window=500):
        left = max(0, peak-window)
        right= min(len(arr), peak+window)
        return left, right

    left_w, right_w = windowaround(snr_white, idx_peak_white, 500)
    t_sub = times[left_w:right_w]

    ax2[0].plot(t_sub, snr_white[left_w:right_w], label='Corr(whitened)', color='C0')
    ax2[0].axvline(times[idx_peak_white], color='C0', linestyle=':')
    ax2[0].axvline(true_tc, color='k', linestyle='-.', label='True $t_c$')
    ax2[0].set_ylabel("Cross-corr amplitude")
    ax2[0].set_title("Toy matched filter (whitened)")

    left_b, right_b = windowaround(snr_bp, idx_peak_bp, 500)
    t_sub2 = times[left_b:right_b]

    ax2[1].plot(t_sub2, snr_bp[left_b:right_b], label='Corr(bp+white)', color='C1')
    ax2[1].axvline(times[idx_peak_bp], color='C1', linestyle=':')
    ax2[1].axvline(true_tc, color='k', linestyle='-.')
    ax2[1].set_ylabel("Cross-corr amplitude")
    ax2[1].set_xlabel("Time [s]")
    ax2[1].set_title("Toy matched filter (band-pass+whitened)")

    ax2[0].legend(loc='upper left')
    ax2[1].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("snr_outputs_example.pdf", bbox_inches='tight')
    plt.close()

    print("Saved: filtered_waveforms_example.pdf and snr_outputs_example.pdf")


if __name__=="__main__":
    main()
