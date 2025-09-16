# Mouse brain timescales from IBL data

Codes for estimating autocorrelation timescales from IBL data. If you use this code for a scientific publication, please cite this paper:    
Shi*, Y. L., Zeraati*, R., The International Brain Laboratory, Levina, A., & Engel, T. A. (2025). Brain-wide organization of intrinsic timescales at single-neuron resolution. bioRxiv, 2025-08. https://doi.org/10.1101/2025.08.30.673281 

We estimated intrinsic timescales of single-neuron activity from the spike-count autocorrelation computed during 10-minute recordings of spontaneous activity at the end of each session. The autocorrelations were computed using the [abcTau package](https://github.com/roxana-zeraati/abcTau).

To estimate the timescales, we used scipy `curve_fit` function to fit the AC shapes with a mixture of exponential decay functions with up to four timescales, and applied the Bayesian information criterion (BIC) to select the best-fitting model. In addition, we required each timescale to contribute at least 1\% to the overall autocorrelation shape; otherwise, we selected the model with fewer timescales, even if a more complex model was preferred by the BIC. This additional constraint ensured that our analysis focused only on timescales that substantially contributed to neural dynamics.

`timescale_utils` contains all the fitting functions, and `fit_acs` is the script that loads the autocorrelation data, runs the fitting algorithm and saves the results.


The main timescales estimation function is `fit_model_comp_BIC`. The output of fitting returned as `selected_model` can be one of the following:
-  0: single exponential decay
-  1: two expoenential decays
-  2: three expoenential decays
-  3: four expoenential decays
-  101: None of the models above could fit the data
-  20x: The $R^2$ (coefficient of determination) of the fitted model was too low (below `R2_thresh`), x is the index of fitted model (see above).

For the selected model, it also returns model parameters as `selected_params` and $R^2$ as `selected_R2`.


## Operating system
- macOS
- Windows
- Linux

## Dependencies
- Python >= 3.11.7
- Numpy >= 1.26.4 
- Scipy >= 1.11.4