import numpy as np
from scipy.optimize import curve_fit



def four_exp(time, tau1, tau2, tau3, tau4, coef1, coef2, coef3, coef4):
    """a mixture of four expoenetial decay functions.

    Parameters
    -----------
    time : 1d array
        time points.
    a : float
        amplitude of autocorrelation at lag 0. 
    tau1 : float
       first timescale.
    tau2 : float
       second timescale.
    tau3 : float
        third timescale.
    tau4 : float
        forth timescale.
    coef1: float
        weight of the first timescale between [0,1]
    coef2: float
        weight of the second timescale between [0,1]
    coef3: float
        weight of the third timescale between [0,1]
    coef4: float
        weight of the third timescale between [0,1]
    
    
    Returns
    -------
    exp_func : 1d array
        decay function.
    """
    exp_func = ((coef1) * np.exp(-time/tau1) + (coef2) * np.exp(-time/tau2) +\
    (coef3) * np.exp(-time/tau3) +  (coef4) * tau4)
    return  exp_func


def triple_exp(time, tau1, tau2, tau3, coef1, coef2, coef3):
    """a triple expoenetial decay function.

    Parameters
    -----------
    time : 1d array
        time points.
    a : float
        amplitude of autocorrelation at lag 0. 
    tau1 : float
       first timescale.
    tau2 : float
       second timescale.
    tau3 : float
        third timescale.
    coef1: float
        weight of the first timescale between [0,1]
    coef2: float
        weight of the second timescale between [0,1]
    coef3: float
        weight of the third timescale between [0,1]
    
    Returns
    -------
    exp_func : 1d array
        decay function.
    """
    exp_func = ((coef1) * np.exp(-time/tau1) + (coef2) * np.exp(-time/tau2) +\
    (coef3) * np.exp(-time/tau3))
    return  exp_func


def double_exp(time, tau1, tau2, coef1, coef2):
    """a double expoenetial decay function.

    Parameters
    -----------
    time : 1d array
        time points.
    a : float
        amplitude of autocorrelation at lag 0. 
    tau1 : float
       first timescale.
    tau2 : float
       second timescale.
    coef1: float
        weight of the first timescale between [0,1]
    coef2: float
        weight of the second timescale between [0,1]
    
    
    Returns
    -------
    exp_func : 1d array
        decay function.
    """
    exp_func = ((coef1) * np.exp(-time/tau1) + (coef2) * np.exp(-time/tau2))
    return  exp_func


def single_exp(time, a, tau):
    """a single expoenetial decay function.

    Parameters
    -----------
    time : 1d array
        time points.
    a : float
        amplitude of autocorrelation at lag 0. 
    tau : float
       timescale.
    
    
    Returns
    -------
    exp_func : 1d array
        decay function.
    """
    exp_func = a * np.exp(-time/tau) 
    return exp_func

def single_exp_mult_oscil(time, a, tau, f):
    """a single multiplicative damped oscillation.

    Parameters
    -----------
    time : 1d array
        time points.
    a : float
        amplitude of autocorrelation at lag 0. 
    tau : float
       timescale.
    f: float
        frequency of oscillation
    
    
    Returns
    -------
    exp_func : 1d array
        decay function.
    """
    exp_func = a * np.cos(2*np.pi*f* time) * np.exp(-time/tau) 
    return exp_func


def single_exp_add_oscil(time, a, tau, f, coeff):
    """a single additive damped oscillation.

    Parameters
    -----------
    time : 1d array
        time points.
    a : float
        amplitude of autocorrelation at lag 0. 
    tau : float
       timescale.
    f: float
        frequency of oscillation
    coeff: float [0,1]
        coefficient of oscillation
    
    Returns
    -------
    exp_func : 1d array
        decay function.
    """
    exp_func = a * coeff * np.cos(2*np.pi*f* time) + a * (1-coeff) * np.exp(-time/tau) 
    return exp_func

def compute_goodness_fit(ydata, yfit, xdata , k):
    """compute R2 and AIC for a given fit.

    Parameters
    -----------
    ydata: 1d array
        data y axis.
    yfit: 1d array
        fit.
    xdata : 1d array
       data x axis.
    k: int
        number of parameters in the fitted model.
    
    Returns
    -------
    AIC : float
        AIC criterion.
    R2: float
        Coefficient of determination.
    """
    RSS =((yfit - ydata)**2).sum()
    n = len(xdata) # number of samples
    v = n - k # degree of freedom
    AIC = 2*k + n*np.log(RSS/n)    
    # AIC = 2*k + n*np.log(RSS/v) # version using X^2
    R2 = 1 - (RSS / (((np.mean(ydata) - ydata)**2).sum()))
    return AIC, R2

def compute_goodness_fit_BIC(ydata, yfit, xdata , k):
    """compute R2 and BIC for a given fit.

    Parameters
    -----------
    ydata: 1d array
        data y axis.
    yfit: 1d array
        fit.
    xdata : 1d array
       data x axis.
    k: int
        number of parameters in the fitted model.
    
    Returns
    -------
    BIC : float
        BIC criterion.
    R2: float
        Coefficient of determination.
    """
    RSS =((yfit - ydata)**2).sum()
    n = len(xdata) # number of samples
    v = n - k # degree of freedom
    BIC = k * np.log(n) + n * np.log(RSS/n) 
    # BIC = k * np.log(n) + n * np.log(RSS/v) # version using X^2    
    R2 = 1 - (RSS / (((np.mean(ydata) - ydata)**2).sum()))
    return BIC, R2

    


def fit_model_comp_BIC(ac, lags, bin_size, max_lag, if_noisiness_exclude = False, lag_exclude = 0.02, thresh_exclude = 0.01,\
                               R2_thresh = 0.5, max_tau = 600, min_tau = 0.005, coef_thresh = 0.001):
    """
    Fitting AC with multiple models, apply exclusion criteria, do model comparison using BIC. 
    Considers a bound on fitted parameters.
    
    Result of model comparison is returned in selected_model:
    0: single exponential decay
    1: two expoenential decays
    2: three expoenential decays
    3: four expoenential decays
    100: AC shape decayed too fast or was too noisy (this is not used in the final version, we use R2 instead)
    101: None of the models above could fit the data
    20x: The R2 of the fitted model was too low (below R2_thresh), x is the index of fitted model (see above).
    
    
    Parameters
    -----------
    ac : 1d array
        autocorrelation.
    lags : 1d array
        time lags.
    bin_size: float
        bin size of psike counts
    max_lag: float
        maximum time lag 
    if_noisiness_exclude: boolean
        If True apply noisiness criterion exclusion.
    lag_exclude: float
        maximum time lag for checking the exclusion criterion.
    thresh_exclude: float
        AC threshold of the exclusion criterion.
    R2_thresh: float [0,1]
        threshold for exclusion based on low coefficient of determination (goodness of fit)
    max_tau: float
        upper bound on estimated timescales
    coef_thresh: float
        minimum value of a coefficient to be considered in model selection

    
    
    Returns
    -------
    selected_model : int
        id of selected model.
    selected_params: list
        list of parameters of the best fitted model.
    selected_R2: float
        Coefficient of determination of the selected model (goodness of fit)
        
    """
    

    # define minimum time lag as the lag with the first decay 
    min_lag_id = np.where(np.diff(ac)<0)[0][1]
    max_lag_id = int(max_lag/bin_size)
    
    xdata = lags[min_lag_id:max_lag_id+1]
    ydata = ac[min_lag_id:max_lag_id+1]/ac[0]
    
    # set initial values
    popt = np.nan
    R2 = np.nan
    
    # excluding noisy ACs
    # check and label noisy ACs # this is not used in final ananlyses
    if_noisy = 0
    lag_exclude_id = int(lag_exclude/bin_size)
    if np.sum(ydata[:lag_exclude_id]<thresh_exclude) >0:
        if_noisy = 1
        
    
    #---------------- model fittings ------------------------#
    BIC_all = []
    params_all = []
    R2_all = []
    
    # 0) fitting one exponential
    try:
        popt, pcov = curve_fit(single_exp, xdata, ydata, maxfev = 2000, bounds=((0,min_tau),(1., max_tau)))
        yfit = single_exp(xdata, *popt)
        
        k = 2 # number of parameters
        BIC, R2 = compute_goodness_fit_BIC(ydata, yfit, xdata , k)
       
    except Exception:
                BIC = 10**5
                pass
    BIC_all.append(BIC)
    params_all.append(popt)
    R2_all.append(R2)
    
    # 1) fitting two exponentials
    try:
        popt, pcov = curve_fit(double_exp, xdata, ydata, maxfev = 2000, bounds=((min_tau,min_tau,0,0),(max_tau, max_tau, 1.0, 1.0)))
        yfit = double_exp(xdata, *popt)

        k = 4 # number of parameters
        BIC, R2 = compute_goodness_fit_BIC(ydata, yfit, xdata , k)
        
        # check if taus are very similar, then exclude from model comparison
        tau1 = popt[0]
        tau2 = popt[1]        
        if abs(tau1 - tau2) < min_tau:
            BIC = 10**5
        # check if one of the coeffcients is very small, then exclude from model comparison
        if np.sum(popt[2:] < coef_thresh):
            BIC = 10**5
    except Exception:
                BIC = 10**5
                pass
    BIC_all.append(BIC)
    params_all.append(popt)
    R2_all.append(R2)
    
    # 2) fitting three exponentials
    try:
        popt, pcov = curve_fit(triple_exp, xdata, ydata, maxfev = 2000, bounds=((min_tau,min_tau,min_tau,0,0,0),(max_tau, max_tau, max_tau, 1.0, 1.0, 1.0)))
        yfit = triple_exp(xdata, *popt)

        k = 6 # number of parameters
        BIC, R2 = compute_goodness_fit_BIC(ydata, yfit, xdata , k)
        
        # check if taus are very similar, then exclude from model comparison
        tau1 = popt[0]
        tau2 = popt[1]
        tau3 = popt[2]
        taus = np.sort(np.array([tau1, tau2, tau3]))
        if np.sum(np.diff(taus) < min_tau) > 0:
            BIC = 10**5
        # check if one of the coeffcients is very small, then exclude from model comparison
        if np.sum(popt[3:] < coef_thresh):
            BIC = 10**5
    except Exception:
                BIC = 10**5
                pass
    BIC_all.append(BIC)
    params_all.append(popt)
    R2_all.append(R2)
    
    # 3) fitting four exponentials
    try:
        popt, pcov = curve_fit(four_exp, xdata, ydata, maxfev = 2000, bounds=((min_tau,min_tau,min_tau,min_tau,0,0,0,0),(max_tau, max_tau, max_tau, max_tau, 1.0, 1.0, 1.0, 1.0)))
        yfit = four_exp(xdata, *popt)

        k = 8 # number of parameters
        BIC, R2 = compute_goodness_fit_BIC(ydata, yfit, xdata , k)
        
        # check if taus are very similar, then exclude from model comparison
        tau1 = popt[0]
        tau2 = popt[1]
        tau3 = popt[2]
        tau4 = popt[3]
        taus = np.sort(np.array([tau1, tau2, tau3, tau4]))
        if np.sum(np.diff(taus) < min_tau) > 0:
            BIC = 10**5
        # check if one of the coeffcients is very small, then exclude from model comparison
        if np.sum(popt[4:] < coef_thresh):
            BIC = 10**5
    except Exception:
                BIC = 10**5
                pass
    BIC_all.append(BIC)
    params_all.append(popt)
    R2_all.append(R2)
    
    # compare models using BIC, best model has minimum BIC
    selected_model = np.argsort(BIC_all)[0]
    selected_params = params_all[selected_model]
    selected_R2 = R2_all[selected_model]
    
    
    # check if R2 is too low
    if selected_R2 < R2_thresh:
        selected_model = int('20' + str(selected_model))
        return selected_model, selected_params, selected_R2, params_all, if_noisy
    
    return selected_model, selected_params, selected_R2, params_all, if_noisy


