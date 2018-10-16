"""This module contains functions to work with signals and noise.

.. module:: noise
    :synopsis: Noise functions for signals.
    
.. moduleauthor:: Marco Guerreiro <marcotulio.guerreiro@gmail.com>

"""
import numpy as np

def awgn(x, snr, out='signal', method='vectorized'):

    """Adds White Gaussian Noise to a signal.

    The noise level is specified as a Signal-to-Noise Ratio (SNR) value.
    The SNR is defined as:
    
    .. math:: SNR = 10\\log_{10}\\left(\\frac{E_x}{E_n}\\right)

    where :math:`E_x` is the signal power and :math:`E_n` is the noise
    power. The noise of a discrete signal can be computed as:

    .. math:: E = \\frac{1}{N}\sum_{k=0}^{N - 1}|x_k|^2
    
    Parameters
    ----------
        x : `np.ndarray`
            Signal, as a vector or column-matrix.
        snr : int, float
            Signal-to-Noise ration.
        out : str, optional
            Output data. If 'signal', the signal `x` plus noise is
            returned. If 'noise', only the noise vector is returned. If
            'both', signal with noise and noise only are returned. Any
            other value defaults to 'signal'.
        method: str, optional
            Method to compute noise vector (matrix) to be introduced in the
            signal. In the 'vectorized' method, the matrix energy is computed
            and used to compute the noise energy. In the 'max_en' method, the
            energy of each column of `x` is computed and only the highest
            value is used to compute the noise energy. The 'vectorized' method
            is used by default.

    Returns
    -------
    `np.ndarray`
        Corrupted signal.

    Raises
    ------
    ValueErrorExpection
        If `method` is not recognized.

    """    
    N = x.shape[0]
    
    # Signal power
    # Ps = np.sum(x**2/N, axis=0)
    if method == 'vectorized':
        Ps = np.sum(x**2/N)
    elif method == 'max_en':
        Ps = np.max(np.sum(x**2/N, axis=0))
    else:
        raise ValueError('method \"' + str(method) + '\" not recognized.')
    
    Psdb = 10*np.log10(Ps)

    # Noise level necessary
    Pn = Psdb - snr

    # Noise vector (or matrix)
    n = np.sqrt(10**(Pn/10))*np.random.normal(0, 1, x.shape)

    if out == 'signal':
        return x + n
    elif out == 'noise':
        return n
    elif out == 'both':
        return x + n, n
    else:
        return x + n
