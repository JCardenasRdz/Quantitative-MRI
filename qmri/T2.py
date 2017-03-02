_author_ = "Julio Cardenas"

import numpy as np

def sim(xdata=np.linspace(0,.5,20),pars=[1.0,0.05]):
    '''
    Simulates T2 signal using an spin echo sequence
    # S = Mz * exp(-TE/T2)
    xdata = np.linspace(0,.5,20)
    p     = pars[1.0,0.05]
    '''
    R2= -1.0 * pars[1]**-1
    return pars[0] * np.exp(R2*xdata)
