"""Noise generation test #2

Procedure
---------
* Generate a random matrix, where all entries are integers.
* Add noise with the pynoise package, using the 'vectorized' method.
* Compute the difference between the original matrix and the matrix corrupted
  with noise, that is, get the pure noise signal.
* Compute the SNR between the original matrix and the obtained matrix.

Output
------
The script prints in the screen the desired SNR and the obtained SNR.

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
m = 100
n = 3

# Bounds
lower_b = -15
upper_b = 15

# Noise
SNR = 30

# --- Signals ---
A = np.random.randint(lower_b, upper_b, (m, n))

# --- Noise ---
An = pynoise.awgn(A, SNR, method='vectorized')
ns = A - An

# --- Check actual SNR ---
Em_dB = 10*np.log10(np.sum(A**2))
En_dB = 10*np.log10(np.sum(ns**2))
SNR_dB = Em_dB - En_dB

print('\nDesired SNR: %.2f dB' % SNR)
print('\nComputed SNR: %.2f dB' % SNR_dB)
