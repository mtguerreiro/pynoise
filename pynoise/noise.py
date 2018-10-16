"""This module contains functions to work with signals and noise.

.. module:: noise
    :synopsis: Noise functions for signals.
    
.. moduleauthor:: Marco Guerreiro <marcotulio.guerreiro@gmail.com>

"""
import numpy as np

def awgn(x, snr, out='signal'):

    """Adds White Gaussian Noise to a signal.

    The noise level is specified as a Signal-to-Noise Ratio (SNR) value.
    The SNR is defined as:
    
    .. math:: SNR = 10\\log_{10}\\left(\\frac{E_x}{E_n}\\right)

    where :math:`E_x` is the signal power and :math:`E_n` is the noise
    power. The noise of a discrete signal can be computed as:

    .. math:: E = \\frac{1}{N}\sum_{k=0}^{N - 1}|x_k|^2
    
    Parameters
    ----------
        x : ``np.ndarray``
            Signal.
        snr : int, float
            Signal-to-Noise ration.
        out : str, optional
            Output data. If 'signal', the signal `x` plus noise is
            returned. If 'noise', only the noise vector is returned. If
            'both', signal with noise and noise only are returned. Any
            other value defaults to 'signal'.

    Returns
    -------
    ``np.ndarray``
        Corrupted signal.
        
    """
    if type(x) is list:
        x = np.array(x)
    
    N = x.shape[0]
    
    # Signal power
    Ps = np.sum(x**2/N, axis=0)
    
    Psdb = 10*np.log10(Ps)

    # Noise level necessary
    Pn = Psdb - snr

    # Noise vector
    n = np.sqrt(10**(Pn/10))*np.random.normal(0, 1, x.shape)

    if out == 'signal':
        return x + n
    elif out == 'noise':
        return n
    elif out == 'both':
        return x + n, n
    else:
        return x + n
