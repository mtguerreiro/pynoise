import numpy as np

def awgn(x, snr, out='signal'):

    """Introduz ruído AWGN em um sinal.

    Args:
        x (np.ndarray): Sinal a ser adicionado ruído.
        snr (int, float): SNR desejado.
        out (str): Tipo de retorno. Se 'signal', retorna o sinal com o
            ruído. Se 'noise', retorna somente o ruído. Se 'both',
            retorna uma lista, em que o primeiro elemento é o sinal com
            o ruído e o segundo elemento é o ruído. Para qualquer outro
            valor, a função retorna o sinal com o ruído.

    Returns:
        Corrupted signal
    """

    if type(x) is list:
        x = np.array(x)
    
    #N = len(x)
    N = x.shape[0]
    
    # First, compute signal power
    #Ps = sum(x**2/N)
    Ps = np.sum(x**2/N, axis=0)
    
    Psdb = 10*np.log10(Ps)

    # Next, we compute how much noise we need
    Pn = Psdb - snr

    # Finally, we compute the noise vector
    n = np.sqrt(10**(Pn/10))*np.random.normal(0, 1, x.shape)

    # If out is y, returns signal plus noise. Else, returns noise only
    if out == 'signal':
        return x + n
    elif out == 'noise':
        return n
    elif out == 'both':
        return x + n, n
    else:
        return x + n

