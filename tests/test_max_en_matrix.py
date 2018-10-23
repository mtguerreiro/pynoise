"""Noise generation test #4

Procedure
---------

* Generate a Nx3 matrix, where:
  * A[:, 0] = 0.8t
  * A[:, 1] = 1.0t
  * A[:, 2] = 1.3t
* Add noise with the pynoise package, using the 'max_en' method.
* Compute the difference between the original matrix and the matrix corrupted
  with noise, that is, get the pure noise signal.
* Compute the SNR between the original matrix and the obtained matrix.

Output
------
The script prints in the screen the obtained SNR for each column. The SNR is
computed based on the signal and noise energy of each column of the original
matrix and the noise matrix, respectively.

Expected results
----------------
Since the third column is the one with highest energy, the noise will be
generated to produce the specified SNR only for that column. All other columns
should have worse SNR, since their energy is weaker compared to the
third column, and the noise energy is the same for all columns.

"""
import numpy as np
import pynoise

import matplotlib.pyplot as plt
plt.ion()

# --- Input ---
N = 1000

# Noise
SNR = 30

# --- Signals ---
t = np.arange(N)

A = np.zeros((N, 3))
A[:, 0] = 0.8*t
A[:, 1] = 1.0*t
A[:, 2] = 1.3*t

# --- Noise ---
An = pynoise.awgn(A, SNR, method='max_en')
ns = A - An

# --- Check actual SNR ---
Em_dB = 10*np.log10(np.sum(A**2, axis=0))
En_dB = 10*np.log10(np.sum(ns**2, axis=0))
SNR_dB = Em_dB - En_dB

print('\nSNR 1st column: %.2f dB' % SNR_dB[0])
print('\nSNR 2nd column: %.2f dB' % SNR_dB[1])
print('\nSNR 3rd column: %.2f dB' % SNR_dB[2])
