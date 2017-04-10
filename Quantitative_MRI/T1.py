import numpy as np
from scipy.optimize import least_squares

"Class definition and methods"

class T1_vtr_signal(object):
    """
    T1 vTR signal class, with the following attributes:
    ---------------------------------------------------
    x: float, ndarray
        Repetition time in seconds (xdata)
    y: float, ndarray
        observed/simulated signal
    yhat: float, ndarray
        clean/fitted signal
    T1 = estimated T1
    Mz = estimated Mz

    """
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
        self.T1 = 0.0
        self.Mz = 0.0
        self.yhat = np.zeros_like(y)

    def add_noise(self,noise):
        self.y = _add_noise(self.y, noise)



"Functions needed to simulate and fit data"

def T1rec(TR, T1 = 2.5, Mz = 1.0, noise = 0.0):
    """
    Description
    ----------
    Simulate Spin Echo Saturation Recovery Experiment

    sig = Mz * (1 - np.exp(-RepTime/T1))

    Parameters
    ----------
    TR: float
        ndarray of repetition times (TR) in seconds
    T1: float, optional
        T1 time in seconds (float)
    Mz: float, optional
        Thermal equilibrium magnetization in in AU
    noise: float, optional
        Standard deviation of Gaussian noise

    Returns
    ----------
    Signal : ndarray of the signal at each TR

    Examples
    ----------
    >>> T1rec(np.linspace(0,10,5))
    array([ 0.        ,  0.63212056,  0.86466472,  0.95021293,  0.98168436])
    >>> T1rec(np.linspace(0,10,5),T1=2.0)
    array([ 0.        ,  0.7134952 ,  0.917915  ,  0.97648225,  0.99326205])
    >>> T1rec(np.linspace(0,10,5),T1=10.0, Mz=100)
    array([  0.        ,  22.11992169,  39.34693403,  52.76334473,  63.21205588])

    """
    signal = Mz * (1 - np.exp(-TR/T1))
    signal = _add_noise(signal, noise)
    return np.squeeze(signal)

def _fit(TR, Yobserved):
    """
    Solve a nonlinear least-squares problem with bounds on the variables.
    Using the module: scipy.optimize.least_squares

    Parameters
    ----------
    TR: float ndarray of epetition times (TR) in seconds
    Yobserved: float ndarray1 of observed signal at each TRtime in seconds (float)
    residual_function:  Object linking to residual function of the form:
                        func(params) = ydata - f(xdata, params)

    Returns
    ----------
    x0_pred: float array of size [2,] where:
    x0_pred[0] = T1 estimate
    x0_pred[1] = Mz estimate

    Ypredicted = float ndarray of predicted signal according to model
    """
    # scale signal
    max_signal = max(Yobserved)
    Yobserved = Yobserved/max_signal

    # Initial guess for T1 = 3.0, and Mz = 1.0
    x0 = np.asarray([3.0, 1.0])
    # Lower bounds
    lb  = [0.1,0.1]; lb = np.asarray(lb)

    # upper bounds
    ub =  [5.0,2.0]; ub = np.asarray(ub)

    # normalize signal

    # run non-linear regression
    R = least_squares(_residual_function,
                          x0,
                          bounds=(0, np.inf),
                          args=(Yobserved, TR, T1rec))

    pars_predicted = [R.x[0],R.x[1]*max_signal]
    pars_predicted = np.asarray(pars_predicted)

    Ypredicted = T1rec(TR,  T1 = pars_predicted[0],
                            Mz = pars_predicted[1])

    return pars_predicted, Ypredicted

def _residual_function(parameters, signal_observed, TR, T1Model):
    """
    Estimates the residuals between and observed signal and the T1 recovery model
    """
    signal_predicted = T1Model(TR, T1=parameters[0], Mz=parameters[1])
    # make sure dimensionalities are correct
    signal_predicted = np.squeeze(signal_predicted)
    signal_observed = np.squeeze(signal_observed)
    # calc and return residuals
    residuals =  signal_observed - signal_predicted
    return residuals

"Utility functions"
def _add_noise(signal, sigma):
    "Add random guassian noise with mean of zero and std of sigma"
    noise = np.random.normal(loc=0.0, scale=sigma, size=signal.shape)
    return noise + signal
