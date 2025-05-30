#!/usr/bin/env python3
"""
Demonstration script using LALSuite to generate a 30+30 Msun CBC waveform at 100 Mpc,
embedding it in Gaussian noise plus multiple line interferences, whitening based on a
separate noise PSD estimate, band-pass + notch filtering, and performing a toy matched
filter cross-correlation.

Plots generated:
  1) waveforms.pdf:
       - 3 subplots: raw, whitened, and filtered (each on separate y-axes)
  2) waveforms_overlay.pdf:
       - A single axis with raw, whitened, and filtered waveforms,
         each offset vertically to avoid compression.
  3) snr_autoscale.pdf:
       - matched filter SNR in 2 subplots (full and zoom ±0.5s), each autoscaled
  4) snr_fixed.pdf:
       - matched filter SNR in 2 subplots (full and zoom), with a fixed y-limit
  5) debug_fft_masks.pdf:
       - debugging plot showing bandpass+notch frequency mask

Usage: python this_script.py
Outputs all PDFs in current directory.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lal
import lalsimulation

def main():
    # -------------------------------
    # 1. Define global parameters
    # -------------------------------
    fs         = 1024.0     # sampling rate (Hz)
    duration   = 8.0        # total data length (s)
    N          = int(duration*fs)
    time_vec   = np.linspace(-duration/2, duration/2, N, endpoint=False)  # from -4s to +4s
    delta_t    = 1.0/fs

    # CBC parameters
    m1_kg      = 30.0 * lal.MSUN_SI
    m2_kg      = 30.0 * lal.MSUN_SI
    distance_m = 100e6 * lal.PC_SI  # 100 Mpc -> stronger signal for demonstration
    spin1x = spin1y = spin1z = 0.0
    spin2x = spin2y = spin2z = 0.0
    inclination= 0.0
    phiRef     = 0.0
    longAscNodes = 0.0
    eccentricity = 0.0
    meanPerAno   = 0.0
    f_lower    = 20.0
    f_ref      = 20.0
    approximant= lalsimulation.IMRPhenomD

    # Interference line frequencies & amplitudes
    line_freqs_amps = [(1.2, 60.0), (0.9, 120.0), (0.7, 180.0), (0.5, 240.0), (0.5, 300.0)]

    # Band-pass range & notches
    bp_low, bp_high = 20.0, 300.0
    line_notches = [60.0, 120.0, 180.0, 240.0, 300.0]

    # -------------------------------
    # 2. Generate waveform with LAL
    # -------------------------------
    hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
        m1_kg, m2_kg,
        spin1x, spin1y, spin1z,
        spin2x, spin2y, spin2z,
        distance_m, inclination, phiRef,
        longAscNodes, eccentricity, meanPerAno,
        delta_t, f_lower, f_ref,
        lal.CreateDict(),
        approximant
    )
    waveform_arr = hp.data.data
    Nsig = len(waveform_arr)

    # Align the waveform so its largest amplitude sample is at t=0 in an 8s record
    peak_idx = np.argmax(np.abs(waveform_arr))
    signal = np.zeros(N)
    start_idx = (N//2) - peak_idx
    end_idx   = start_idx + Nsig
    if start_idx < 0 or end_idx> N:
        raise ValueError("Waveform too long or offset to embed at center.")
    signal[start_idx:end_idx] = waveform_arr

    # Normalize signal to unit amplitude
    signal = signal / np.max(np.abs(signal))

    # -------------------------------
    # 3. Make synthetic noise
    #    (a) separate noise to measure PSD
    #    (b) main noise + lines in the data
    # -------------------------------
    np.random.seed(42)
    noise_for_psd = np.random.normal(0, 1.0, N)  # measure "noise only" amplitude
    data_noise = np.random.normal(0, 1.0, N)
    # add lines
    for (amp, freq) in line_freqs_amps:
        data_noise += amp * np.sin(2*np.pi*freq*time_vec)

    # combine noise + signal
    data_raw = 10 * signal + data_noise

    # measure PSD from noise_for_psd
    noise_fft_psd = np.fft.rfft(noise_for_psd)
    noise_asd = np.abs(noise_fft_psd)
    noise_asd = np.where(noise_asd>1e-12, noise_asd, 1e-12)

    # -------------------------------
    # 4. Whiten data & signal
    # -------------------------------
    data_fft = np.fft.rfft(data_raw)
    white_data_fft = data_fft / noise_asd
    white_data = np.fft.irfft(white_data_fft, n=N)

    signal_fft = np.fft.rfft(signal)
    white_signal_fft = signal_fft / noise_asd
    white_signal = np.fft.irfft(white_signal_fft, n=N)

    # -------------------------------
    # 5. band-pass + notch masks
    # -------------------------------
    freq_bins = np.fft.rfftfreq(N, d=delta_t)
    band_mask = (freq_bins >= bp_low) & (freq_bins <= bp_high)

    notch_mask = np.ones_like(band_mask, dtype=bool)
    for f0 in line_notches:
        idx_line = np.argmin(np.abs(freq_bins - f0))
        notch_mask[idx_line] = False

    final_mask = band_mask & notch_mask

    # debug mask plot
    plt.figure(figsize=(8,4))
    plt.plot(freq_bins, band_mask*1.0, label="Band-pass mask")
    plt.plot(freq_bins, notch_mask*0.8, label="Notch mask")
    plt.plot(freq_bins, final_mask*0.6, label="Combined final mask")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Mask value")
    plt.legend()
    plt.title("Frequency mask debug (bandpass + notches)")
    plt.tight_layout()
    plt.savefig("debug_fft_masks.pdf")
    plt.close()

    # Apply final_mask to whitened data & signal
    wdf_filt = white_data_fft.copy()
    wdf_filt[~final_mask] = 0.0
    filtered_data = np.fft.irfft(wdf_filt, n=N)

    wsf_filt = white_signal_fft.copy()
    wsf_filt[~final_mask] = 0.0
    filtered_signal = np.fft.irfft(wsf_filt, n=N)

    # -------------------------------
    # 6. Matched filter: cross-correlation in freq domain
    # -------------------------------
    def matched_filter_toy(data_t, templ_t):
        L = len(data_t)
        N2 = 2 * L
        df = np.fft.rfft(data_t, n=N2)
        tf = np.fft.rfft(templ_t, n=N2)
        corrf = df * np.conjugate(tf)
        corr  = np.fft.irfft(corrf, n=N2)
        # shift so lag=0 is at index=(L-1)
        corr = np.roll(corr, - (L -1))
        return corr[:(2*L-1)]

    corr = matched_filter_toy(filtered_data, filtered_signal)
    # normalize
    sigpow = np.sum(np.abs(filtered_signal)**2)
    norm_factor = np.sqrt(sigpow) if sigpow>1e-12 else 1e-12
    snr_arr = corr / norm_factor

    # build time axis for correlation
    lag_array = np.arange(-(N - 1), (N - 1)+1)
    lag_time  = lag_array * delta_t
    mask_snr  = (lag_time>=-4.0) & (lag_time<=4.0)
    snr_time  = lag_time[mask_snr]
    snr_vals  = snr_arr[mask_snr]

    # zoom window ±0.5
    zoom_win = 0.5
    mask_zoom = (snr_time>=-zoom_win) & (snr_time<=zoom_win)
    snr_t_zoom= snr_time[mask_zoom]
    snr_v_zoom= snr_vals[mask_zoom]

    # Print the peak SNR, time
    peak_idx_snr = np.argmax(np.abs(snr_vals))
    print("Peak correlation index in [-4..+4]:",
          f"t={snr_time[peak_idx_snr]:.4f}s,  SNR={snr_vals[peak_idx_snr]:.3f}")

    # -------------------------------
    # 7. PLOT WAVEFORMS => separate subplots
    # -------------------------------
    fig_w, axs_w = plt.subplots(3,1, figsize=(8,9), sharex=True)
    fig_w.suptitle("Time-Domain Waveforms (Separate Axes)", y=0.95)

    axs_w[0].plot(time_vec, data_raw, color='gray')
    axs_w[0].set_title("Raw data (noise + signal)")
    axs_w[0].set_ylabel("Amplitude")

    axs_w[1].plot(time_vec, white_data, color='C0')
    axs_w[1].set_title("Whitened data")
    axs_w[1].set_ylabel("Whitened")

    axs_w[2].plot(time_vec, filtered_data, color='C1')
    axs_w[2].set_title("Whitened + Bandpass + Notch")
    axs_w[2].set_ylabel("Filtered")
    axs_w[2].set_xlabel("Time [s] (0=peak)")

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig("waveforms.pdf")
    plt.close()

    # (7b) Additional overlay plot with vertical offsets
    fig_overlay = plt.figure(figsize=(10,5))
    plt.title("Overlay: Raw vs. Whitened vs. Filtered (w/ vertical offsets)")
    plt.xlabel("Time [s] (0=peak)")
    # We'll compute offsets: the standard deviation of each series
    # so each series is shifted up by some multiple of the previous std
    off1 = 0.0
    off2 = np.mean(np.abs(data_raw))*3.0
    off3 = off2 + np.mean(np.abs(white_data))*3.0

    plt.plot(time_vec, data_raw + off1, label="Raw data", color="gray")
    plt.plot(time_vec, white_data + off2, label="Whitened", color="C0")
    plt.plot(time_vec, filtered_data + off3, label="Filtered", color="C1")

    plt.ylabel("Amplitude + offset")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("waveforms_overlay.pdf")
    plt.close()

    # -------------------------------
    # 8. PLOT SNR => AUTOSCALE and FIXED
    # -------------------------------
    # (A) Autoscale
    figA, (axA1, axA2) = plt.subplots(2,1, figsize=(8,6))
    axA1.plot(snr_time, snr_vals, color='C2')
    axA1.set_title("Matched Filter SNR: Full [-4..+4] (Autoscale)")
    axA1.set_ylabel("Correlation amplitude")

    axA2.plot(snr_t_zoom, snr_v_zoom, color='C2')
    axA2.set_title("Matched Filter SNR: Zoom ±0.5s (Autoscale)")
    axA2.set_xlabel("Time [s] from t=0")
    axA2.set_ylabel("Correlation amplitude")

    plt.tight_layout()
    plt.savefig("snr_autoscale.pdf")
    plt.close()

    # (B) Fixed scale
    figB, (axB1, axB2) = plt.subplots(2,1, figsize=(8,6))

    axB1.plot(snr_time, snr_vals, color='C2')
    axB1.set_title("Matched Filter SNR: Full [-4..+4] (Fixed scale)")
    axB1.set_ylabel("Correlation amplitude")

    axB2.plot(snr_t_zoom, snr_v_zoom, color='C2')
    axB2.set_title("Matched Filter SNR: Zoom ±0.5s (Fixed scale)")
    axB2.set_xlabel("Time [s] from t=0")
    axB2.set_ylabel("Correlation amplitude")

    # Determine global max for fixed scale
    max_snr = np.max(np.abs(snr_vals))
    y_lim = max_snr * 1.2
    axB1.set_ylim(-y_lim, y_lim)
    axB2.set_ylim(-y_lim, y_lim)

    plt.tight_layout()
    plt.savefig("snr_fixed.pdf")
    plt.close()

    print("Finished! Created waveforms.pdf, waveforms_overlay.pdf, snr_autoscale.pdf, snr_fixed.pdf, debug_fft_masks.pdf")

if __name__=="__main__":
    main()
