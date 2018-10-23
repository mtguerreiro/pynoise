"""Noise generation test #3

Procedure
---------

* Generate a ramp signal of the form :math:`x = at + b` and a cossine signal
  of the form :math:`x = d + a*cos(2\pi ft)`.
* Add noise with the pynoise package, using the 'max_en' method.
* Compute the difference between the original signal and the signal corrupted
  with noise, that is, get the pure noise signal.
* Compute the SNR between the original signal and the obtained noise.

Output
------
The script prints in the screen the resulting SNR for both signals.

Expected results
----------------
The obtained SNR should match the desired SNR. It will never be equal, since
the noise is randomly generated and fluctuations will cause noise signals
with more or less energy. Nevertheless, the obtained SNR should be close
to the specified SNR.

"""
import numpy as np
import pynoise

# --- Input ---
N = 10000

# Ramp signal
# Angular (a) and linear (b) coefficients
a_ramp = 1
b_ramp = 1

# Cossine signal
# Sampling frequency, signal frequency, AC amplitude and DC component
fs_cos = 1e4
f_cos = 10
a_cos = 2
d_cos = -0.5

# Noise
SNR = 30

# --- Signals ---
t_ramp = np.arange(N)
x_ramp = a_ramp*t_ramp + b_ramp

t_cos = (1/fs_cos)*np.arange(N)
x_cos = d_cos + a_cos*np.cos(2*np.pi*f_cos*t_cos)

# --- Noise ---
xn_ramp = pynoise.awgn(x_ramp, SNR, method='max_en')
n_ramp = x_ramp - xn_ramp

xn_cos = pynoise.awgn(x_cos, SNR, method='max_en')
n_cos = x_cos - xn_cos

# --- Check actual SNR ---
print('\nDesired SNR: %.2f' % SNR)

# Ramp signal
Es_ramp_dB = 10*np.log10(np.sum(x_ramp**2))
En_ramp_dB = 10*np.log10(np.sum(n_ramp**2))
SNR_ramp_dB = Es_ramp_dB - En_ramp_dB
print('\nRamp signal SNR: %.2f dB' % SNR_ramp_dB)

# Cossine signal
Es_cos_dB = 10*np.log10(np.sum(x_cos**2))
En_cos_dB = 10*np.log10(np.sum(n_cos**2))
SNR_cos_dB = Es_cos_dB - En_cos_dB
print('\nCossine signal SNR: %.2f dB' % SNR_cos_dB)
