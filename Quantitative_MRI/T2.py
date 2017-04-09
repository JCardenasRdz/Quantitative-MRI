def T2decay(TE = xdata, T2 = 0.5, Mz = 100):
    """
    Simulate t2 decay signal

    Input
    -----
    TE = vector of length N of echo times in seconds (float)
    T2 = T2 time in seconds (float)
    Mz = Thermal equilibrium in AU (float)

    Output
    -----
    Signal = T2 decay signal of length N
    """
    signal = Mz * np.exp(-TE/T2)

    return signal
    

def add_noise(sig,sigma):
    """
    
    add random Gaussian noise
    
    Output
    ------
    return sig = sig + np.random.normal(scale=sigma, size=sig.shape)
    
    """
    return sig + np.random.normal(scale=sigma, size=sig.shape)


    
    
    
    