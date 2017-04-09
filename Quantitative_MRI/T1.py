import numpy as np
from scipy.optimize import least_squares

def T1rec(RepTime, T1 = 2.5, Mz = 1):
    """
    Simulate t2 decay signal
    sig = Mz * (1 - np.exp(-RepTime/T1))
    
    Input
    -----
    RepTime = vector of length N of TR times in seconds (float)
    T1 = T1 time in seconds (float)
    Mz = Thermal equilibrium in AU (float)

    Output
    -----
    Signal = T1 recovery signal of length N
    """
    sig = Mz * (1 - np.exp(-RepTime/T1))

    return sig
    
def _residuals(pars,Yobserved, xdata,func):
    predicted_data = func(xdata,pars[0], pars[1])
    residuals = predicted_data - Yobserved
    return residuals
    #kwargs={}
    
    

def fit(x_data, Yobserved):
    x0 = np.array([1,3])
    x0_pred = least_squares(_residuals,
                          x0,
                          bounds=(0, np.inf),
                          args=(Yobserved, x_data, T1rec))
    return x0_pred

def add_noise(sig_,sigma):
    """
    
    add random Gaussian noise
    
    Output
    ------
    return sig = sig + np.random.normal(scale=sigma, size=sig.shape)
    
    """
    return sig_ + np.random.normal(scale=sigma, size=sig_.shape)


        
    