import numpy as np
import pynoise

import matplotlib
import matplotlib.pyplot as plt

# --- Input ---
# Signal
f = 60
fs = 12000
cycles = 5
dc = 1
ac = 1

# Noise
SNR = 40

# --- Definitions ---
# Time
t = np.arange(0, cycles/f, 1/fs)

# Signal
s = dc + ac*np.sin(2*np.pi*f*t)

# Corrupted signal
sn, n = pynoise.awgn(s, SNR, out='both')

# --- Check actual SNR ---
# Signal power, dB
Ps = 10*np.log10(sum(s**2/len(s)))

# Noise power, dB
Pn = 10*np.log10(sum(n**2/len(n)))

# Actual SNR
SNRa = Ps - Pn
print('Desired SNR: ', SNR)
print('Actual SNR: ', SNRa)

# --- Difference ---
e = s - sn

# --- FFT ---
N = len(n)
Xr = np.fft.fft(n)
fx = [fs/N*fxj for fxj in np.arange(0,N/2)]
Xrmag = [abs(xr/N) for xr in Xr[0:int(N/2)]]
Xrmag = [Xrmag[0]] + [2*xr for xr in Xrmag[1:]]

# --- Plots ---
# Signal
plt.figure(figsize=(10,7))
plt.plot(t/1e-3, s, lw=2, label=r'Sinal')
plt.plot(t/1e-3, sn, lw=2, label=r'Sinal corrompido')
plt.xlabel(r'Tempo (ms)')
plt.ylabel(r'Tensão (V)')
plt.title(r'Sinal corrompido por ruído')
plt.legend()
plt.grid()

# Noise hist
plt.figure(figsize=(10,7))
plt.subplot(2, 1, 1)
plt.hist(n/1e-3, bins=20)
plt.xlabel('Tensão (mv)')
plt.ylabel('Frequência')
plt.title(r'Histograma do sinal de ruído')
plt.grid()

# Noise spectra
plt.subplot(2, 1, 2)
markerline, stemlines, baseline = plt.stem(fx, Xrmag)
plt.setp(baseline, visible=False)
plt.xlabel(r'$f$ (Hz)')
plt.ylabel(r'$|X_r|$')
plt.title(r'Sinal de ruído no espectro de frequência')
plt.grid()
plt.tight_layout()
plt.savefig('noise_specta.png', bbox_inches='tight')
plt.show()
