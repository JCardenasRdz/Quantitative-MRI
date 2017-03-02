_author_ = "Julio Cardenas"
import numpy as np

def sim(xdata=np.linspace(0,10,20),pars=[1.0,3]):
    '''
    Simulates T1 signal using an variabel TR sequence
    # S = Mz * (1-exp(-TR/T1))
    xdata = np.linspace(0,10,20)
    p     = pars=[1.0,3]
    '''
    R1= -1.0 * pars[1]**-1
    return pars[0] * (1 - np.exp(R1*xdata) )
