import numpy as np
import os
from os.path import expanduser
import pickle
import timescale_utils as tu

# setting for parallel processing
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "4" 


# load data
data_path =  expanduser("~") +'/research/data/projects/mouse_timescales/ac/'

# loading the autocorrelation data and setting the parameters
ac_all_sing = np.load(data_path + 'timescale_10sec_ac.npy') # #unit=27960 * #time bin=200
bin_size = 0.005 # bin size in sec
num_units, num_bins = np.shape(ac_all_sing)
lags = np.arange(0, num_bins*bin_size , bin_size) # in sec

# fit ACs and estimate taus
max_lag = 5 # maximum time lag for fitting the autocorrelation in sec
print('max lag = ', max_lag)
max_tau = 600 # maximum of timescale fitting range in sec (based on the length of time-series)
min_tau = 0 # minimum of timescale fitting range in sec 


# fitting the data and recording the results
selected_model_all = [] # selected model after model comparison with BIC
selected_params_all = [] # parameters of the selected model
selected_R2_all = [] # R2 of the selected model
params_all = [] # parameters of all fitted models
for n in range(num_units):
    print(n)
    ac = ac_all_sing[n]
    selected_model, selected_params, selected_R2, params_allModels, if_noisy = \
    tu.fit_model_comp_BIC(ac, lags, bin_size, max_lag, R2_thresh = 0.5, max_tau = max_tau, min_tau = min_tau, coef_thresh = 0.01)
    
    selected_model_all.append(selected_model)
    selected_params_all.append(selected_params)
    selected_R2_all.append(selected_R2)
    params_all.append(params_allModels)
    
    
# making a dictionay and saving the results
save_data = {'selected models': selected_model_all,
             'selected params': selected_params_all,
            'selected R2': selected_R2_all,
            'all params': params_all}


with open(data_path + 'fitted_models_single_unit_BIC_maxlag' + str(max_lag) + '.pkl', 'wb') as f:
            pickle.dump(save_data, f)
