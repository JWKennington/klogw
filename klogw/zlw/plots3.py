#!/usr/bin/env python3
"""
Unified demonstration script using LALSuite to generate a 30+30 Msun CBC waveform at 400 Mpc,
adding synthetic Gaussian noise plus 60-Hz harmonics line interference, applying whitening,
bandpass+notch filtering, and performing a toy matched filter via cross-correlation in freq domain.

Generates 4 PDF figures:

  1) 'waveforms.pdf': Time-domain waveforms in separate subplots:
       (a) raw data (signal+noise),
       (b) whitened data,
       (c) whitened+bandpass+notch filtered data.
     Each subplot has an independent y-scale.

  2) 'snr_autoscale.pdf': Matched filter SNR (full time) and zoomed around t=0, with autoscaled y-limits.

  3) 'snr_fixed.pdf': The same SNR time-series, but both subplots share a common y-limit,
     making it easy to compare absolute amplitude in the zoom vs. full view.

  4) 'debug_fft_masks.pdf': (Optional) If you want to visualize the bandpass+notch masks in frequency domain.

Time is in seconds, from -4 to +4, with t=0 at merger peak. This script is purely illustrative.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # For headless environments; remove if you want interactive
import matplotlib.pyplot as plt

import lal
import lalsimulation

########################################
# 1) GENERATE CBC WAVEFORM WITH LALSUITE
########################################
fs       = 1024.0           # sample rate (Hz)
delta_t  = 1.0 / fs
duration = 8.0              # total duration (s)
N        = int(duration * fs)
time_vec = np.linspace(-duration/2, duration/2, N, endpoint=False)  # from -4s to +4s

# CBC parameters
m1_kg   = 30.0 * lal.MSUN_SI
m2_kg   = 30.0 * lal.MSUN_SI
dist_m  = 400e6 * lal.PC_SI   # 400 Mpc in meters
spin1x = spin1y = spin1z = 0.0
spin2x = spin2y = spin2z = 0.0
inclination = 0.0
phiRef     = 0.0
longAscNodes = 0.0
ecc        = 0.0
meanAno    = 0.0
f_lower    = 20.0
f_ref      = 20.0
approximant= lalsimulation.IMRPhenomD

# Create TD waveform with sampling interval delta_t
hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
    m1_kg, m2_kg,
    spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
    dist_m, inclination, phiRef,
    longAscNodes, ecc, meanAno,
    delta_t, f_lower, f_ref,
    lal.CreateDict(),
    approximant
)
waveform_array = hp.data.data
Nsig = len(waveform_array)

# We center the waveform peak at t=0 in our 8s record
# Find the maximum amplitude sample index in the generated waveform
peak_idx = np.argmax(np.abs(waveform_array))
# We'll embed the waveform so that waveform's peak aligns with index = N//2
signal = np.zeros(N)
start_index = (N // 2) - peak_idx
end_index   = start_index + Nsig
# If there's any out-of-bounds issue, we can clamp:
if start_index < 0:
    raise ValueError("Waveform is too long to fit in the 8s record with peak at center.")
if end_index > N:
    raise ValueError("Waveform is too long; it doesn't fit in the 8s record.")
signal[start_index:end_index] = waveform_array

########################################
# 2) SYNTHETIC NOISE + 60-HZ LINES + SIGNAL
########################################
np.random.seed(42)
noise = np.random.normal(0, 1.0, N)  # Gaussian noise
# Add line interference at multiples of 60 Hz
for (amp, freq) in [(3.0, 60.0), (2.0, 120.0), (1.5, 180.0), (1.0, 240.0), (0.8, 300.0)]:
    noise += amp * np.sin(2.0 * np.pi * freq * time_vec)

# Combine
data_raw = signal + noise

########################################
# 3) WHITENING (TOY METHOD)
########################################
# We'll approximate the amplitude of 'noise' in freq domain and use that for data
noise_fft  = np.fft.rfft(noise)
noise_asd  = np.abs(noise_fft)
noise_asd  = np.where(noise_asd > 1e-12, noise_asd, 1e-12)  # avoid zero

data_fft = np.fft.rfft(data_raw)
white_data_fft = data_fft / noise_asd
white_data = np.fft.irfft(white_data_fft, n=N)

signal_fft = np.fft.rfft(signal)
white_signal_fft = signal_fft / noise_asd
white_signal = np.fft.irfft(white_signal_fft, n=N)

########################################
# 4) BAND-PASS + NOTCH ON WHITENED DATA
########################################
freq_bins = np.fft.rfftfreq(N, d=delta_t)  # freq axis for rFFT
band_mask = (freq_bins >= 20.0) & (freq_bins <= 300.0)

# Notch out 60-Hz multiples if within 20–300
notch_mask = np.ones_like(band_mask, dtype=bool)
for f0 in [60.0, 120.0, 180.0, 240.0, 300.0]:
    bin_idx = np.argmin(np.abs(freq_bins - f0))
    if (0 <= bin_idx < len(notch_mask)):
        notch_mask[bin_idx] = False

final_mask = band_mask & notch_mask  # True => keep freq, False => zero out

white_data_fft_filt = white_data_fft.copy()
white_data_fft_filt[~final_mask] = 0.0
filtered_data = np.fft.irfft(white_data_fft_filt, n=N)

white_signal_fft_filt = white_signal_fft.copy()
white_signal_fft_filt[~final_mask] = 0.0
filtered_signal = np.fft.irfft(white_signal_fft_filt, n=N)

# OPTIONAL: debug plot of freq mask
plt.figure(figsize=(8,4))
plt.plot(freq_bins, band_mask*1.0, label='Band-pass mask')
plt.plot(freq_bins, notch_mask*0.8, label='Notch mask')
plt.plot(freq_bins, final_mask*0.6, label='Final combined mask')
plt.xlabel("Frequency [Hz]")
plt.ylabel("Mask value (1=pass,0=stop)")
plt.title("Debug: Frequency masks for band-pass & notch")
plt.legend()
plt.tight_layout()
plt.savefig("debug_fft_masks.pdf")
plt.close()

########################################
# 5) MATCHED FILTER (TOY CROSS-CORRELATION)
########################################
# We'll do freq-domain multiplication of data and conj(template), then iFFT => correlation
def matched_filter_toy(data_t, templ_t):
    """Compute cross-correlation (like matched filter SNR) between data and template.
       data_t and templ_t are time-domain arrays, both length N.
       We'll do zero-padding to length 2N for linear correlation, then find the peak near index ~N."""
    L = len(data_t)
    N2 = 2 * L
    data_fft_2   = np.fft.rfft(data_t, n=N2)
    templ_fft_2  = np.fft.rfft(templ_t, n=N2)
    corr_freq_2  = data_fft_2 * np.conjugate(templ_fft_2)
    corr_time_2  = np.fft.irfft(corr_freq_2, n=N2)
    # correlation length = 2N-1. The zero-lag is at index=0 in typical numpy correlation, but we want to interpret peak near L-1
    # We'll shift so that index L => 0 lag, i.e. we rotate the array
    corr_time_2  = np.roll(corr_time_2, - (L - 1))  # shift so that 'lag=0' is in middle
    return corr_time_2[:L*2-1]  # length 2N-1

corr_white = matched_filter_toy(filtered_data, filtered_signal)
# We can normalize by sqrt of (sum(template^2)) so it becomes an SNR-like amplitude
templ_norm = np.sqrt(np.sum(np.abs(filtered_signal)**2))
snr = corr_white / (templ_norm if templ_norm>0 else 1e-12)

# Build a time axis for the correlation from - (N-1) to + (N-1) samples
lag_array = np.arange(-(N - 1), (N - 1))
lag_time  = lag_array * delta_t  # in seconds
# We want the portion that corresponds to the data window (~ -4 to +4)
# The zero-lag is near index=0 in snr array => that means t=0 is in the middle
# We'll clamp the time to [-4, +4]
mask_snr = (lag_time >= -4.0) & (lag_time <= 4.0)
snr_time = lag_time[mask_snr]
snr_vals = snr[mask_snr]

# For a zoom around t=0 +/- 0.5
zoom_win = 0.5
mask_zoom = (snr_time >= -zoom_win) & (snr_time <= zoom_win)
snr_time_zoom = snr_time[mask_zoom]
snr_vals_zoom = snr_vals[mask_zoom]

########################################
# 6) PLOT WAVEFORMS (SEPARATE SUBPLOTS, NO SHARED Y)
########################################
fig_w, axs_w = plt.subplots(3, 1, figsize=(8,9), sharex=True)
fig_w.suptitle("Time-Domain Waveforms with Separate Axes", fontsize=14, y=0.95)

axs_w[0].plot(time_vec, data_raw, color='gray')
axs_w[0].set_title("Raw data (signal + noise)")
axs_w[0].set_ylabel("Amplitude")

axs_w[1].plot(time_vec, white_data, color='C0')
axs_w[1].set_title("Whitened data")
axs_w[1].set_ylabel("Whitened units")

axs_w[2].plot(time_vec, filtered_data, color='C1')
axs_w[2].set_title("Whitened + bandpassed + notched")
axs_w[2].set_ylabel("Filtered units")
axs_w[2].set_xlabel("Time [s], 0=merger peak")

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("waveforms.pdf")
plt.close()

########################################
# 7) PLOT MATCHED FILTER SNR: AUTOSCALE & FIXED
########################################

# (A) AUTOSCALE
fig_snr_auto, (axA_full, axA_zoom) = plt.subplots(2,1, figsize=(8,6), sharex=False)
axA_full.plot(snr_time, snr_vals, color='C2')
axA_full.set_title("Matched Filter SNR: Full Duration (Autoscale)")
axA_full.set_ylabel("SNR(?)")
axA_full.set_xlabel("Time [s] from t=0 (merger)")

axA_zoom.plot(snr_time_zoom, snr_vals_zoom, color='C2')
axA_zoom.set_title("Matched Filter SNR: Zoom ±0.5s (Autoscale)")
axA_zoom.set_ylabel("SNR")
axA_zoom.set_xlabel("Time [s]")
plt.tight_layout()
plt.savefig("snr_autoscale.pdf")
plt.close()

# (B) FIXED Y-AXIS
fig_snr_fix, (axF_full, axF_zoom) = plt.subplots(2,1, figsize=(8,6), sharex=False)
axF_full.plot(snr_time, snr_vals, color='C2')
axF_full.set_title("Matched Filter SNR: Full Duration (Fixed scale)")
axF_full.set_ylabel("SNR")

axF_zoom.plot(snr_time_zoom, snr_vals_zoom, color='C2')
axF_zoom.set_title("Matched Filter SNR: Zoom ±0.5s (Fixed scale)")
axF_zoom.set_ylabel("SNR")
axF_zoom.set_xlabel("Time [s] from t=0 (merger)")

# Determine global max for snr_vals:
snr_peak = np.max(np.abs(snr_vals))
ylim = snr_peak * 1.2
axF_full.set_ylim(-ylim, ylim)
axF_zoom.set_ylim(-ylim, ylim)

plt.tight_layout()
plt.savefig("snr_fixed.pdf")
plt.close()

print("Done. Created: waveforms.pdf, snr_autoscale.pdf, snr_fixed.pdf, debug_fft_masks.pdf.")
