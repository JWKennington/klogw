"""Below are Python scripts (using numpy, scipy.signal, and matplotlib) that generate the pole-zero plots, impulse responses, and frequency responses for each of the systems discussed above. Each code block corresponds to one filter example.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Low-pass filter parameters

# # Analog: H(s) = 1/(s+1)

num_an = [1]            # numerator coefficients of 1
den_an = [1, 1]         # denominator coefficients for s+1

# # Discrete: H(z) = (1-α)/(1 - α z^{-1}), α=0.9

alpha = 0.9
b_dig = [1-alpha]       # [b0] (numerator in z^-1 form)
a_dig = [1, -alpha]     # [1, -α] (denominator in z^-1 form)

# Compute poles and zeros
# Analog poles, zeros:

zeros_an = np.roots(num_an)    # should be [] (no finite zeros)
poles_an = np.roots(den_an)

# Discrete poles, zeros:
# To get finite zeros, form polynomial in z: multiply by z^(deg denom)

deg_d = len(a_dig)-1  # denom degree
deg_b = len(b_dig)-1  # numer degree
deg = int(max(deg_d, deg_b))

# Numerator polynomial in z:
poly_num = [0]*(deg+1); poly_den = [0]*(deg+1)
for i, coef in enumerate(b_dig):
    poly_num[deg - i] = coef
for i,coef in enumerate(a_dig):
    poly_den[deg - i] = coef
zeros_d = np.roots(poly_num)
poles_d = np.roots(poly_den)

# # Pole-zero plot

fig, ax = plt.subplots(1, 2, figsize=(8,4))
# Analog PZ

ax[0].scatter(np.real(zeros_an), np.imag(zeros_an), marker="o", facecolors="none", edgecolors="b", s=50, label="Zero")
ax[0].scatter(np.real(poles_an), np.imag(poles_an), marker="x", color="r", s=50, label="Pole")
ax[0].axhline(0,color="k", linewidth=0.5); ax[0].axvline(0,color="k", linewidth=0.5)
ax[0].set_title("Analog Low-pass PZ"); ax[0].set_xlabel("Re(s)"); ax[0].set_ylabel("Im(s)"); ax[0].legend()

# Discrete PZ
# Unit circle for reference

theta = np.linspace(0, 2*np.pi, 400); uc = np.exp(1j*theta)
ax[1].plot(np.real(uc), np.imag(uc), "k-", linewidth=0.5)
ax[1].scatter(np.real(zeros_d), np.imag(zeros_d), marker="o", facecolors="none", edgecolors="b", s=50)
ax[1].scatter(np.real(poles_d), np.imag(poles_d), marker="x", color="r", s=50)
ax[1].axhline(0,color="k", linewidth=0.5); ax[1].axvline(0,color="k", linewidth=0.5)
ax[1].set_title("Digital Low-pass PZ"); ax[1].set_xlabel("Re(z)"); ax[1].set_ylabel("Im(z)")
plt.tight_layout(); plt.savefig("lowpass_pz.pdf"); plt.close()

# # Impulse responses
# Analog impulse (use inverse Laplace via scipy.signal.impulse)

sys_an = signal.TransferFunction(num_an, den_an)   # continuous system
T_an, y_an = signal.impulse(sys_an, T=np.linspace(0, 5, 500))

# Discrete impulse (direct convolution)

N = 50
x = np.zeros(N); x[0] = 1
y_d = signal.lfilter(b_dig, a_dig, x)

fig, ax = plt.subplots(1, 2, figsize=(8,3))
ax[0].plot(T_an, y_an, "b")
ax[0].set_title("Analog Low-pass $h(t) = e^{-t}$"); ax[0].set_xlabel("t"); ax[0].set_ylabel("h(t)")
ax[1].stem(range(N), y_d, basefmt=" ",
           # use_line_collection=True,
           )
ax[1].set_title("Digital Low-pass impulse response"); ax[1].set_xlabel("n"); ax[1].set_ylabel("h[n]")
plt.tight_layout(); plt.savefig("lowpass_impulse.pdf"); plt.close()

# Frequency response

w_an = np.linspace(0, 10, 1000)
w_d, H_d = signal.freqz(b_dig, a_dig, worN=1024)
w_d = w_d  # already in radians

_, H_an = signal.freqs(num_an, den_an, worN=w_an)

# Magnitude and phase (analog)

mag_an = 20*np.log10(np.abs(H_an))
phase_an = np.unwrap(np.angle(H_an))*180/np.pi

# Magnitude and phase (digital)

mag_d = 20*np.log10(np.abs(H_d))
phase_d = np.unwrap(np.angle(H_d))*180/np.pi

fig, ax = plt.subplots(2, 2, figsize=(8,6))
ax[0,0].plot(w_an, np.abs(H_an))
ax[0,0].set_title("Analog Low-pass |H(jw)|"); ax[0,0].set_xlabel("ω (rad/s)"); ax[0,0].set_ylabel("Magnitude")
ax[1,0].plot(w_an, phase_an)
ax[1,0].set_title("Analog Low-pass phase"); ax[1,0].set_xlabel("ω (rad/s)"); ax[1,0].set_ylabel("Phase (deg)")
ax[0,1].plot(w_d, np.abs(H_d))
ax[0,1].set_title("Digital Low-pass |H(e^{jΩ})|"); ax[0,1].set_xlabel("Ω (rad)"); ax[0,1].set_ylabel("Magnitude")
ax[1,1].plot(w_d, phase_d)
ax[1,1].set_title("Digital Low-pass phase"); ax[1,1].set_xlabel("Ω (rad)"); ax[1,1].set_ylabel("Phase (deg)")
plt.tight_layout(); plt.savefig("lowpass_freq.pdf"); plt.close()

###############################
# High-pass filter parameters
# # Analog: H(s) = s/(s+1)

num_an = [1, 0]         # numerator s (coeffs for s + 0)
den_an = [1, 1]         # denominator s + 1

# # Discrete: H(z) = α (1 - z^{-1})/(1 - α z^{-1}), α=0.9

alpha = 0.9
b_dig = [alpha, -alpha]  # [α, -α] for α - α z^{-1}
a_dig = [1, -alpha]      # [1, -α]

# Compute poles and zeros

zeros_an = np.roots(num_an)     # should be s=0 zero
poles_an = np.roots(den_an)     # pole at -1
deg_d = len(a_dig)-1; deg_b = len(b_dig)-1; deg = max(deg_d, deg_b)
poly_num = [0]*(deg+1); poly_den = [0]*(deg+1)
for i,coef in enumerate(b_dig):
    poly_num[deg - i] = coef
for i,coef in enumerate(a_dig):
    poly_den[deg - i] = coef
zeros_d = np.roots(poly_num)
poles_d = np.roots(poly_den)

# # Pole-zero plot

fig, ax = plt.subplots(1, 2, figsize=(8,4))

# Analog

ax[0].scatter(np.real(zeros_an), np.imag(zeros_an), marker="o", facecolors="none", edgecolors="b", s=60)
ax[0].scatter(np.real(poles_an), np.imag(poles_an), marker="x", color="r", s=60)
ax[0].axhline(0,color="k", linewidth=0.5); ax[0].axvline(0,color="k", linewidth=0.5)
ax[0].set_title("Analog High-pass PZ"); ax[0].set_xlabel("Re(s)"); ax[0].set_ylabel("Im(s)")

# Discrete

theta = np.linspace(0, 2*np.pi, 400); uc = np.exp(1j*theta)
ax[1].plot(np.real(uc), np.imag(uc), "k-", linewidth=0.5)
ax[1].scatter(np.real(zeros_d), np.imag(zeros_d), marker="o", facecolors="none", edgecolors="b", s=60)
ax[1].scatter(np.real(poles_d), np.imag(poles_d), marker="x", color="r", s=60)
ax[1].axhline(0,color="k", linewidth=0.5); ax[1].axvline(0,color="k", linewidth=0.5)
ax[1].set_title("Digital High-pass PZ"); ax[1].set_xlabel("Re(z)"); ax[1].set_ylabel("Im(z)")
plt.tight_layout(); plt.savefig("highpass_pz.pdf"); plt.close()

# # Impulse responses

sys_an = signal.TransferFunction(num_an, den_an)
T_an, y_an = signal.impulse(sys_an, T=np.linspace(0, 5, 500))
N = 50
x = np.zeros(N); x[0] = 1
y_d = signal.lfilter(b_dig, a_dig, x)

fig, ax = plt.subplots(1, 2, figsize=(8,3))
ax[0].plot(T_an, y_an, "b")
ax[0].set_title(r"Analog High-pass $h(t)=\delta(t) - e^{-t}$"); ax[0].set_xlabel("t"); ax[0].set_ylabel("h(t)")
ax[1].stem(range(N), y_d, basefmt=" ",
           # use_line_collection=True,
           )
ax[1].set_title("Digital High-pass impulse"); ax[1].set_xlabel("n"); ax[1].set_ylabel("h[n]")
plt.tight_layout(); plt.savefig("highpass_impulse.pdf"); plt.close()

# # Frequency responses

w_an = np.linspace(0, 10, 1000)
w_d, H_d = signal.freqz(b_dig, a_dig, worN=1024)
_, H_an = signal.freqs(num_an, den_an, worN=w_an)
mag_an = np.abs(H_an); phase_an = np.unwrap(np.angle(H_an))*180/np.pi
mag_d = np.abs(H_d); phase_d = np.unwrap(np.angle(H_d))*180/np.pi

fig, ax = plt.subplots(2, 2, figsize=(8,6))
ax[0,0].plot(w_an, mag_an)
ax[0,0].set_title("Analog High-pass |H(jw)|"); ax[0,0].set_xlabel("ω"); ax[0,0].set_ylabel("Magnitude")
ax[1,0].plot(w_an, phase_an)
ax[1,0].set_title("Analog High-pass phase"); ax[1,0].set_xlabel("ω"); ax[1,0].set_ylabel("Phase (deg)")
ax[0,1].plot(w_d, mag_d)
ax[0,1].set_title("Digital High-pass |H(e^{jΩ})|"); ax[0,1].set_xlabel("Ω"); ax[0,1].set_ylabel("Magnitude")
ax[1,1].plot(w_d, phase_d)
ax[1,1].set_title("Digital High-pass phase"); ax[1,1].set_xlabel("Ω"); ax[1,1].set_ylabel("Phase (deg)")
plt.tight_layout(); plt.savefig("highpass_freq.pdf"); plt.close()

#################################################
# Band-pass filter parameters

# Analog: H(s) = s/(s^2 + 2 ζ ω0 s + ω0^2), choose ω0=5, ζ=0.3

w0 = 5.0; zeta = 0.3
num_an = [1, 0]                         # numerator s^1 + 0
den_an = [1, 2*zeta*w0, w0**2]          # s^2 + 2 ζ ω0 s + ω0^2

# # Discrete: H(z) = (1 - z^{-2})/(1 - 2 r cosΩ0 z^{-1} + r^2 z^{-2})

Omega0 = 0.25*np.pi; r = 0.8

# numerator: 1 - z^-2 -> [1, 0, -1]

b_dig = [1, 0, -1]
a_dig = [1, -2*r*np.cos(Omega0), r**2]

# Compute poles and zeros

zeros_an = np.roots(num_an)    # zero at s=0
poles_an = np.roots(den_an)
deg_d = len(a_dig)-1; deg_b = len(b_dig)-1; deg = max(deg_d, deg_b)
poly_num = [0]*(deg+1); poly_den = [0]*(deg+1)
for i,coef in enumerate(b_dig):
    poly_num[deg - i] = coef
for i,coef in enumerate(a_dig):
    poly_den[deg - i] = coef
zeros_d = np.roots(poly_num)
poles_d = np.roots(poly_den)

# # Pole-zero plot

fig, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].scatter(np.real(zeros_an), np.imag(zeros_an), marker="o", facecolors="none", edgecolors="b", s=60)
ax[0].scatter(np.real(poles_an), np.imag(poles_an), marker="x", color="r", s=60)
ax[0].axhline(0,color="k", linewidth=0.5); ax[0].axvline(0,color="k", linewidth=0.5)
ax[0].set_title("Analog Band-pass PZ"); ax[0].set_xlabel("Re(s)"); ax[0].set_ylabel("Im(s)")
theta = np.linspace(0, 2*np.pi, 400); uc = np.exp(1j*theta)
ax[1].plot(np.real(uc), np.imag(uc), "k-", linewidth=0.5)
ax[1].scatter(np.real(zeros_d), np.imag(zeros_d), marker="o", facecolors="none", edgecolors="b", s=60)
ax[1].scatter(np.real(poles_d), np.imag(poles_d), marker="x", color="r", s=60)
ax[1].axhline(0,color="k", linewidth=0.5); ax[1].axvline(0,color="k", linewidth=0.5)
ax[1].set_title("Digital Band-pass PZ"); ax[1].set_xlabel("Re(z)"); ax[1].set_ylabel("Im(z)")
plt.tight_layout(); plt.savefig("bandpass_pz.pdf"); plt.close()

# # Impulse responses

sys_an = signal.TransferFunction(num_an, den_an)
T_an, y_an = signal.impulse(sys_an, T=np.linspace(0, 10, 1000))
N = 50
x = np.zeros(N); x[0] = 1
y_d = signal.lfilter(b_dig, a_dig, x)
fig, ax = plt.subplots(1, 2, figsize=(8,3))
ax[0].plot(T_an, y_an, "b")
ax[0].set_title("Analog Band-pass impulse"); ax[0].set_xlabel("t"); ax[0].set_ylabel("h(t)")
ax[1].stem(range(N), y_d, basefmt=" ")
ax[1].set_title("Digital Band-pass impulse"); ax[1].set_xlabel("n"); ax[1].set_ylabel("h[n]")
plt.tight_layout(); plt.savefig("bandpass_impulse.pdf"); plt.close()

# # Frequency responses

w_an = np.linspace(0, 15, 1000)
w_d, H_d = signal.freqz(b_dig, a_dig, worN=1024)
_, H_an = signal.freqs(num_an, den_an, worN=w_an)
mag_an = np.abs(H_an); phase_an = np.unwrap(np.angle(H_an))*180/np.pi
mag_d = np.abs(H_d); phase_d = np.unwrap(np.angle(H_d))*180/np.pi
fig, ax = plt.subplots(2, 2, figsize=(8,6))
ax[0,0].plot(w_an, mag_an)
ax[0,0].set_title("Analog Band-pass |H(jw)|"); ax[0,0].set_xlabel("ω"); ax[0,0].set_ylabel("Magnitude")
ax[1,0].plot(w_an, phase_an)
ax[1,0].set_title("Analog Band-pass phase"); ax[1,0].set_xlabel("ω"); ax[1,0].set_ylabel("Phase (deg)")
ax[0,1].plot(w_d, mag_d)
ax[0,1].set_title("Digital Band-pass |H(e^{jΩ})|"); ax[0,1].set_xlabel("Ω"); ax[0,1].set_ylabel("Magnitude")
ax[1,1].plot(w_d, phase_d)
ax[1,1].set_title("Digital Band-pass phase"); ax[1,1].set_xlabel("Ω"); ax[1,1].set_ylabel("Phase (deg)")
plt.tight_layout(); plt.savefig("bandpass_freq.pdf"); plt.close()

#################################################
# Notch filter parameters

# # Analog: H(s) = (s^2 + ω0^2)/(s^2 + 2 ζ ω0 s + ω0^2)

w0 = 15.0; zeta = 0.1
num_an = [1, 0, w0*2]
den_an = [1, 2*zeta*w0, w0*2]

# # Discrete: H(z) = (1 - 2 cosΩ0 z^{-1} + z^{-2})/(1 - 2 r cosΩ0 z^{-1} + r^2 z^{-2})

Omega0 = 0.4*np.pi; r = 0.95
b_dig = [1, -2*np.cos(Omega0), 1]
a_dig = [1, -2*r*np.cos(Omega0), r**2]

# Poles and zeros

zeros_an = np.roots(num_an)
poles_an = np.roots(den_an)
deg_d = len(a_dig)-1; deg_b = len(b_dig)-1; deg = max(deg_d, deg_b)
poly_num = [0]*(deg+1); poly_den = [0]*(deg+1)
for i,coef in enumerate(b_dig):
    poly_num[deg - i] = coef
for i,coef in enumerate(a_dig):
    poly_den[deg - i] = coef
zeros_d = np.roots(poly_num)
poles_d = np.roots(poly_den)

# Pole-zero plot

fig, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].scatter(np.real(zeros_an), np.imag(zeros_an), marker="o", facecolors="none", edgecolors="b", s=60)
ax[0].scatter(np.real(poles_an), np.imag(poles_an), marker="x", color="r", s=60)
ax[0].axhline(0,color="k", linewidth=0.5); ax[0].axvline(0,color="k", linewidth=0.5)
ax[0].set_title("Analog Notch PZ"); ax[0].set_xlabel("Re(s)"); ax[0].set_ylabel("Im(s)")
theta = np.linspace(0, 2*np.pi, 400); uc = np.exp(1j*theta)
ax[1].plot(np.real(uc), np.imag(uc), "k-", linewidth=0.5)
ax[1].scatter(np.real(zeros_d), np.imag(zeros_d), marker="o", facecolors="none", edgecolors="b", s=60)
ax[1].scatter(np.real(poles_d), np.imag(poles_d), marker="x", color="r", s=60)
ax[1].axhline(0,color="k", linewidth=0.5); ax[1].axvline(0,color="k", linewidth=0.5)
ax[1].set_title("Digital Notch PZ"); ax[1].set_xlabel("Re(z)"); ax[1].set_ylabel("Im(z)")
plt.tight_layout(); plt.savefig("notch_pz.pdf"); plt.close()

# Impulse responses

sys_an = signal.TransferFunction(num_an, den_an)
T_an, y_an = signal.impulse(sys_an, T=np.linspace(0, 2, 1000))
N = 200
x = np.zeros(N); x[0] = 1
y_d = signal.lfilter(b_dig, a_dig, x)
fig, ax = plt.subplots(1, 2, figsize=(8,3))
ax[0].plot(T_an, y_an, "b")
ax[0].set_title("Analog Notch impulse"); ax[0].set_xlabel("t"); ax[0].set_ylabel("h(t)")
ax[1].stem(range(N), y_d, basefmt=" ")
ax[1].set_title("Digital Notch impulse"); ax[1].set_xlabel("n"); ax[1].set_ylabel("h[n]")
plt.tight_layout(); plt.savefig("notch_impulse.pdf"); plt.close()

# Frequency responses

w_an = np.linspace(0, 30, 1000)
w_d, H_d = signal.freqz(b_dig, a_dig, worN=2048)
_, H_an = signal.freqs(num_an, den_an, worN=w_an)
mag_an = np.abs(H_an); phase_an = np.unwrap(np.angle(H_an))*180/np.pi
mag_d = np.abs(H_d); phase_d = np.unwrap(np.angle(H_d))*180/np.pi
fig, ax = plt.subplots(2, 2, figsize=(8,6))
ax[0,0].plot(w_an, mag_an)
ax[0,0].set_title("Analog Notch |H(jw)|"); ax[0,0].set_xlabel("ω"); ax[0,0].set_ylabel("Magnitude")
ax[1,0].plot(w_an, phase_an)
ax[1,0].set_title("Analog Notch phase"); ax[1,0].set_xlabel("ω"); ax[1,0].set_ylabel("Phase (deg)")
ax[0,1].plot(w_d, mag_d)
ax[0,1].set_title("Digital Notch |H(e^{jΩ})|"); ax[0,1].set_xlabel("Ω"); ax[0,1].set_ylabel("Magnitude")
ax[1,1].plot(w_d, phase_d)
ax[1,1].set_title("Digital Notch phase"); ax[1,1].set_xlabel("Ω"); ax[1,1].set_ylabel("Phase (deg)")
plt.tight_layout(); plt.savefig("notch_freq.pdf"); plt.close()


# \begin{lstlisting}[language=Python, caption=Resonator filter (analog & digital) plots]

# Resonator filter parameters

# Analog: H(s) = ω0^2/(s^2 + 2 ζ ω0 s + ω0^2)

w0 = 5.0; zeta = 0.05
num_an = [0, 0, w0*2]
den_an = [1, 2*zeta*w0, w0*2]

# Discrete: H(z) = 1/(1 - 2 r cosΩ0 z^{-1} + r^2 z^{-2})

Omega0 = 0.333*np.pi; r = 0.98
b_dig = [1]                     # numerator 1
a_dig = [1, -2*r*np.cos(Omega0), r**2]

# Poles and zeros

zeros_an = np.roots(num_an)    # (two zeros at s=0 from double pole at infinity)
poles_an = np.roots(den_an)
deg_d = len(a_dig)-1; deg_b = len(b_dig)-1; deg = max(deg_d, deg_b)
poly_num = [0]*(deg+1); poly_den = [0]*(deg+1)
for i,coef in enumerate(b_dig):
    poly_num[deg - i] = coef
for i,coef in enumerate(a_dig):
    poly_den[deg - i] = coef
zeros_d = np.roots(poly_num)
poles_d = np.roots(poly_den)

# Pole-zero plot

fig, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].scatter(np.real(zeros_an), np.imag(zeros_an), marker="o", facecolors="none", edgecolors="b", s=60)
ax[0].scatter(np.real(poles_an), np.imag(poles_an), marker="x", color="r", s=60)
ax[0].axhline(0,color="k", linewidth=0.5); ax[0].axvline(0,color="k", linewidth=0.5)
ax[0].set_title("Analog Resonator PZ"); ax[0].set_xlabel("Re(s)"); ax[0].set_ylabel("Im(s)")
theta = np.linspace(0, 2*np.pi, 400); uc = np.exp(1j*theta)
ax[1].plot(np.real(uc), np.imag(uc), "k-", linewidth=0.5)
ax[1].scatter(np.real(zeros_d), np.imag(zeros_d), marker="o", facecolors="none", edgecolors="b", s=60)
ax[1].scatter(np.real(poles_d), np.imag(poles_d), marker="x", color="r", s=60)
ax[1].axhline(0,color="k", linewidth=0.5); ax[1].axvline(0,color="k", linewidth=0.5)
ax[1].set_title("Digital Resonator PZ"); ax[1].set_xlabel("Re(z)"); ax[1].set_ylabel("Im(z)")
plt.tight_layout(); plt.savefig("resonator_pz.pdf"); plt.close()

# Impulse responses

sys_an = signal.TransferFunction(num_an, den_an)
T_an, y_an = signal.impulse(sys_an, T=np.linspace(0, 50, 1000))
N = 200
x = np.zeros(N); x[0] = 1
y_d = signal.lfilter(b_dig, a_dig, x)
fig, ax = plt.subplots(1, 2, figsize=(8,3))
ax[0].plot(T_an, y_an, "b")
ax[0].set_title("Analog Resonator impulse"); ax[0].set_xlabel("t"); ax[0].set_ylabel("h(t)")
ax[1].stem(range(N), y_d, basefmt=" ")
ax[1].set_title("Digital Resonator impulse"); ax[1].set_xlabel("n"); ax[1].set_ylabel("h[n]")
plt.tight_layout(); plt.savefig("resonator_impulse.pdf"); plt.close()

# Frequency responses

w_an = np.linspace(0, 10, 1000)
w_d, H_d = signal.freqz(b_dig, a_dig, worN=4096)
_, H_an = signal.freqs(num_an, den_an, worN=w_an)
mag_an = np.abs(H_an); phase_an = np.unwrap(np.angle(H_an))*180/np.pi
mag_d = np.abs(H_d); phase_d = np.unwrap(np.angle(H_d))*180/np.pi
fig, ax = plt.subplots(2, 2, figsize=(8,6))
ax[0,0].plot(w_an, mag_an)
ax[0,0].set_title("Analog Resonator |H(jw)|"); ax[0,0].set_xlabel("ω"); ax[0,0].set_ylabel("Magnitude")
ax[1,0].plot(w_an, phase_an)
ax[1,0].set_title("Analog Resonator phase"); ax[1,0].set_xlabel("ω"); ax[1,0].set_ylabel("Phase (deg)")
ax[0,1].plot(w_d, mag_d)
ax[0,1].set_title("Digital Resonator |H(e^{jΩ})|"); ax[0,1].set_xlabel("Ω"); ax[0,1].set_ylabel("Magnitude")
ax[1,1].plot(w_d, phase_d)
ax[1,1].set_title("Digital Resonator phase"); ax[1,1].set_xlabel("Ω"); ax[1,1].set_ylabel("Phase (deg)")
plt.tight_layout(); plt.savefig("resonator_freq.pdf"); plt.close()


# \begin{lstlisting}[language=Python, caption=Composite filter (low-pass + notch) plots]

# Composite filter (low-pass with notch) parameters

# Analog: H(s) = (ω1) * (s^2 + ω0^2)/[ (s+ω1)(s^2 + 2 ζ ω0 s + ω0^2) ]

w0 = 7.0; zeta = 0.1; w1 = 20.0
num_an = [w1, 0, w1*w0**2]   # ω1(s^2 + ω0^2)
den_an = [1, 2*zeta*w0 + w1, w0*2 + 2*zeta*w0*w1, w0*2 * w1]

# Discrete: cascade H_lp and H_notch

alpha = 0.9
b_lp = [1-alpha]; a_lp = [1, -alpha]
Omega0 = 0.4*np.pi; r = 0.95
b_notch = [1, -2*np.cos(Omega0), 1]; a_notch = [1, -2*r*np.cos(Omega0), r**2]

# Convolve polynomials to cascade:

b_dig = np.polynomial.polynomial.polyfromroots(np.roots(b_lp).tolist() + np.roots(b_notch).tolist())
a_dig = np.polynomial.polynomial.polyfromroots(np.roots(a_lp).tolist() + np.roots(a_notch).tolist())
b_dig = np.real_if_close(b_dig).tolist(); a_dig = np.real_if_close(a_dig).tolist()

# Poles and zeros

zeros_an = np.roots(num_an)
poles_an = np.roots(den_an)
deg_d = len(a_dig)-1; deg_b = len(b_dig)-1; deg = max(deg_d, deg_b)
poly_num = [0]*(deg+1); poly_den = [0]*(deg+1)
for i,coef in enumerate(b_dig):
    poly_num[deg - i] = coef
for i,coef in enumerate(a_dig):
    poly_den[deg - i] = coef
zeros_d = np.roots(poly_num)
poles_d = np.roots(poly_den)

# # Pole-zero plot

fig, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].scatter(np.real(zeros_an), np.imag(zeros_an), marker="o", facecolors="none", edgecolors="b", s=60)
ax[0].scatter(np.real(poles_an), np.imag(poles_an), marker="x", color="r", s=60)
ax[0].axhline(0,color="k", linewidth=0.5); ax[0].axvline(0,color="k", linewidth=0.5)
ax[0].set_title("Analog Composite PZ"); ax[0].set_xlabel("Re(s)"); ax[0].set_ylabel("Im(s)")
theta = np.linspace(0, 2*np.pi, 400); uc = np.exp(1j*theta)
ax[1].plot(np.real(uc), np.imag(uc), "k-", linewidth=0.5)
ax[1].scatter(np.real(zeros_d), np.imag(zeros_d), marker="o", facecolors="none", edgecolors="b", s=60)
ax[1].scatter(np.real(poles_d), np.imag(poles_d), marker="x", color="r", s=60)
ax[1].axhline(0,color="k", linewidth=0.5); ax[1].axvline(0,color="k", linewidth=0.5)
ax[1].set_title("Digital Composite PZ"); ax[1].set_xlabel("Re(z)"); ax[1].set_ylabel("Im(z)")
plt.tight_layout(); plt.savefig("composite_pz.pdf"); plt.close()

# Frequency responses

w_an = np.linspace(0, 40, 1000)
w_d, H_d = signal.freqz(b_dig, a_dig, worN=4096)
_, H_an = signal.freqs(num_an, den_an, worN=w_an)
mag_an = np.abs(H_an); phase_an = np.unwrap(np.angle(H_an))*180/np.pi
mag_d = np.abs(H_d); phase_d = np.unwrap(np.angle(H_d))*180/np.pi
fig, ax = plt.subplots(2, 2, figsize=(8,6))
ax[0,0].plot(w_an, mag_an)
ax[0,0].set_title("Analog Composite |H(jw)|"); ax[0,0].set_xlabel("ω"); ax[0,0].set_ylabel("Magnitude")
ax[1,0].plot(w_an, phase_an)
ax[1,0].set_title("Analog Composite phase"); ax[1,0].set_xlabel("ω"); ax[1,0].set_ylabel("Phase (deg)")
ax[0,1].plot(w_d, mag_d)
ax[0,1].set_title("Digital Composite |H(e^{jΩ})|"); ax[0,1].set_xlabel("Ω"); ax[0,1].set_ylabel("Magnitude")
ax[1,1].plot(w_d, phase_d)
ax[1,1].set_title("Digital Composite phase"); ax[1,1].set_xlabel("Ω"); ax[1,1].set_ylabel("Phase (deg)")
plt.tight_layout(); plt.savefig("composite_freq.pdf"); plt.close()

