import numpy as np

def awgn(x, snr, out='y'):
    N = len(x)
    # First, compute signal power
    Ps = sum(x**2/N)
    Psdb = 10*np.log10(Ps)

    # Next, we compute how much noise we need
    Pn = Psdb - snr

    # Finally, we compute the noise vector
    n = np.sqrt(10**(Pn/10))*np.random.normal(0, 1, N)

    # If out is y, returns signal plus noise. Else, returns noise only
    if out is 'y':
        return x + n
    elif out is 'both':
        return x + n, n
    else:
        return n

