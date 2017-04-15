import numpy as np
from scipy.optimize import least_squares as lsq
from .models import T1_vTR
from .models import _residuals

"Class definition and methods"

class T1_vtr_signal(object):
    """

    T1 vTR signal class, with the following:
    ----------------------------------------
    Atrributes
    ----------
    x: float, ndarray
        Repetition time in seconds (xdata)
    yraw: float, ndarray
            original signal before being processed
    yhat: float, ndarray
            predicted signal after curve fitting
    fitted_pars: Result of curve fitting with
    T1 = estimated T1
    Mz = estimated Mz

    Functions
    ----------
    fit: Perform non-linear least-squares curve fitting to estimate T1 and Mz
    """

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.yraw= y
        self.yhat = np.zeros_like(y)
        self.model = T1_vTR
        self.fitted_pars = 0.0

    def fit(self):
        ydata = self.yraw / self.yraw[0]
        x0 = np.array([1,1])
        R = lsq(_residuals, x0, args=(self.x, self.model, ydata))
        self.fitted_pars = return R.x

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
                          args=(Yobserved, TR, T1_sat_rec))

    pars_predicted = [R.x[0],R.x[1]*max_signal]
    pars_predicted = np.asarray(pars_predicted)

    Ypredicted = T1_sat_rec(TR,  T1 = pars_predicted[0],
                            Mz = pars_predicted[1])

    return pars_predicted, Ypredicted

"Utility functions"

def _add_noise(signal, sigma):
    "Add random guassian noise with mean of zero and std of sigma"
    noise = np.random.normal(loc=0.0, scale=sigma, size=signal.shape)
    return noise + signal
