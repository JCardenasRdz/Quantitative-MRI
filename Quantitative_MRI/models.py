import numpy as np

def T1_vTR(TR, *args):
    """
    Description
    ----------
    Simulate Spin Echo Saturation Recovery Experiment

    sig = Mz * (1 - np.exp(-RepTime/T1))

    Parameters
    ----------
    TR: float
        ndarray of repetition times (TR) in seconds
    *args: positional arguments in the following order:
        T1: float, optional
            T1 time in seconds (float)
        Mz: float, optional
            Thermal equilibrium magnetization in in AU
    Returns
    ----------
    Signal : ndarray of the signal at each TR

    Examples
    ----------
    >>> import numpy as np
    >>> x = np.linspace(0,10,11)
    >>> from Quantitative_MRI import models
    >>> models.T1_vTR(x,3, 2)
    array([ 0.        ,  0.56693738,  0.97316576,  1.26424112,  1.47280572,
        1.62224879,  1.72932943,  1.80605606,  1.8610331 ,  1.90042586, 1.92865201])
    """
    T1= args[0]
    Mz= args[1]
    return Mz * (1-np.exp(-TR/T1))

def _residuals(pars, xdata, model_function, observed_data):
    """
    Estimate the residuals
    -----
    Args:
        pars (ndarray):  parameters used in model_function
        xdata (ndarray): Independent variable in model_function
        model_function:  function of the form: model_function(xdata,*pars)
        observed_data :  experimental data used in the fitting

    -----
    Returns:
        residuals (ndarray): residuals of ( observed_data - model_function(xdata,*pars) )
    """
    yhat = model_function(xdata, *pars)
    return observed_data - yhat
