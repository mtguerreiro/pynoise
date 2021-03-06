"""
noise module
============

This module contains functions to work with signals and noise.

.. module:: noise
    :synopsis: Noise functions for signals.
    
.. moduleauthor:: Marco Guerreiro <marcotulio.guerreiro@gmail.com>

"""
import numpy as np


def awgn(x, snr, out='signal', method='vectorized', axis=0):
    r"""Adds White Gaussian Noise to a signal.

    The noise level is specified as a Signal-to-Noise Ratio (SNR) value,
    which relates to signal-to-noise energy or power.

    The SNR between a signal :math:`x` and a noise :math:`n` is defined as:

    .. math::

        \text{SNR} = 10\log\left(\frac{E_x}{E_n}\right),

    where :math:`E_x` is the energy of the signal :math:`x` and
    :math:`E_n` is the energy of the signal :math:`n`.
    
    Parameters
    ----------
    x : `np.ndarray`
        Signal, as a vector or column-matrix.
        
    snr : int, float
        Signal-to-Noise ration.
        
    out : str, optional
        Output data. If 'signal', the signal `x` plus noise is returned. If
        'noise', only the noise vector is returned. If 'both', signal with
        noise and noise only are returned. Any other value defaults to 'signal'.
        
    method : str, optional
        Method to compute noise vector (matrix) to be introduced in the
        signal. In the 'vectorized' method, the matrix energy is computed and
        used to compute the noise energy. In the 'max_en' method, the energy
        of each column of `x` is computed and only the highest value is used
        to compute the noise energy. In the 'axial' method, each column of the
        array is considered as a signal. The 'vectorized' method is used by
        default.
        
    axis : int, optional
        When ``method`` is 'max_en' or 'axis', informs which axis to consider
        as signal. The dafault is 0.

    Returns
    -------
    `np.ndarray`
        Corrupted signal.

    Raises
    ------
    ValueErrorExpection
        If `method` is not recognized.

    Example
    --------
    Add 30 dB of white gaussian noise to the ramp signal
    :math:`x(t) = t`:

    .. plot:: pyplots/noise_ex_app.py
        :include-source:
        :scale: 80
    
    """    
    # Signal power
    if method == 'vectorized':
        N = x.size
        Ps = np.sum(x ** 2 / N)

    elif method == 'max_en':
        N = x.shape[axis]
        Ps = np.max(np.sum(x ** 2 / N, axis=axis))

    elif method == 'axial':
        N = x.shape[axis]
        Ps = np.sum(x ** 2 / N, axis=axis)
        
    else:
        raise ValueError('method \"' + str(method) + '\" not recognized.')

    # Signal power, in dB
    Psdb = 10 * np.log10(Ps)

    # Noise level necessary
    Pn = Psdb - snr

    # Noise vector (or matrix)
    n = np.sqrt(10 ** (Pn / 10)) * np.random.normal(0, 1, x.shape)

    if out == 'signal':
        return x + n
    elif out == 'noise':
        return n
    elif out == 'both':
        return x + n, n
    else:
        return x + n
