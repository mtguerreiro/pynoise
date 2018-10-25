import numpy as np
import pynoise
import matplotlib.pyplot as plt

# --- Signal ---
t = np.arange(0, 1, 0.01)

x = t
xn = pynoise.awgn(x, 30)

# --- Plots ---
plt.figure(figsize=(10,6))
plt.plot(t, x, label='Original signal')
plt.plot(t, xn,  label='Corrupted signal')
plt.grid()
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.show()
