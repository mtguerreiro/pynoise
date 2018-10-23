import numpy as np
import pynoise

# --- Input ---
m = 10
n = 3

# Noise
SNR = 20

# --- Definitions ---
# Matrix
A = np.random.randint(-10, 10, (m, n))

# Corrupted signal
An, ns = pynoise.awgn(A, SNR, out='both', method='vectorized')

# --- Check actual SNR ---
# Signal power, dB
Ps = 10*np.log10(np.sum(A**2))

# Noise power, dB
Pn = 10*np.log10(np.sum(ns**2))

# Actual SNR
SNRa = Ps - Pn
print('Desired SNR: ', SNR)
print('Actual SNR: ', SNRa)
