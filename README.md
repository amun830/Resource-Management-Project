# Resource-Management-Project   

10/09/2021  
####################    Resource Consent Project [Contamination in the Onehunga Aquifer]    ####################  
ENGSCI263 - Group 11    
-   These functions and scripts are used to aid our investigation into the effect of extraction from the Onehunga Aquifer 
    on the pressure and copper concentration, and hence help us make an educated recommendation to the Auckland Regional 
    Council regarding the outcome of Watercare’s consent application.

##### Description #####
-   We created our LPM of the aquifer pressure and copper concentration (fitted the optimal values of unknown parameters)
    by minimising the residual sum of squares.
    
-   From this, we then considered five different "what-if" scenarios:
    -   Stop exraction from the Onehunga Aquifer.
    -   Decrease extraction rate (to 5ML/day).
    -   Decrease extraction rate less (to 10ML/day).
    -   Continue operating at current rate (20ML/day).
    -   Double extraction rate (to 40 ML/day).

-   We then conducted uncertanty analysis by creating parameter posterior distributions, and sampling from these to
    generate sets of paramaters that fit the data well. 

-   Using each set of generated parameters, we forecasted (modelled into the future) for each senario, then plotted this. 

-   These plots allow us to visualise the changes in pressure and copper concentration under each scenario, with uncertainty
    accounted for (and hence help us make a recommendation).

##### How to Use #####
-   First pull the "Resource-Management-Project" from the git repository.
-   Navigate to /Code and Data Files/main.py. Note that all the code is in the "Code and Data Files" folder. 
-   Read the header comments and decide which sections you would like to run.
    -   Switch section 'if' statement to False if you do not wish to run.
-   Wait 20-40 seconds and inspect figure outputs.
    - All figures in report produced, except the one in Lab1Plot.py (see below)

##### Data File Summary #####
-   ac_cu.csv, historic concentration measurements of copper in the aquifer (1985-2015, 5 year intervals, [mg/L]).
-   ac_p.csv, historic pressure measurements of pressure in the aquifer (1985-2016, biannual intervals, [MPa]).
-   ac_q.csv, historic extraction rates of aquifer operation (1985-2018, annual intervals, [10^6 L/day]).

##### Code File Summary (Python) #####
-   main.py
    1. Creates benchmarking plots.
    2. Calibrates model (must be completed for following sections)  
        a. First calibration method     
        b. Second (improved) calibration method.    
    3. Generates forecast plots of future senario.
    4. Conducts uncertainty analysis (generates parameter sets)
    5. Generates plots of future senario predictions, with uncertancy.

-   Copper.py
    1. ode_cu(), returns the derivative dC/dt at time, t, for given parameters.
    2. solve_ode_cu(), solves the Cu concentration ODE numerically (Emproved Euler method).
    3. plot_aquifer_cu(), solves the Cu ODE part of the LPM and plots it over the data.
    4. evaluate_copper(), copper solution helper function for use in scipy.optimize.curve_fit.

-   Pressure.py
    1. ode_pressure(), returns the derivative dP/dt at time, t, for given parameters.
    2. solve_ode_pressure(), solves the pressure ODE numerically (Improved Euler method).
    3. plot_aquifer_pressure(), solves the pressure ODE part of the LPM and plots it over the data.
    4. evaluate_pressure(), pressure solution helper function for use in scipy.optimize.curve_fit.

-   test_solver.py
    1. Asserts our pressure ODE solvers are working correctly (4 tests)
    2. Asserts our copper concentration solvers are working correctly (5 tests).
    Execute the following command in terminal to run unit tests: pytest -v

-   Model_Use_Func.py
    1. plot_aquifer_model(), Solution Plotter Function.
    2. solve_lpm(), Lumped Parameter Model Solver Function.
    3. plot_aquifer_forecast_uncertainty(), Plots forcasted LPM with uncertainty.

-   Helper_Func.py
    1. Defines get_parameter_set(), which combines and arranges parameters into correct arangement.      
        1.1    type == "theta_all", returns list of all calibrated model parameters        
        1.2    type == "cu_all", returns list of all copper model parameters.        
        1.3    type == "split", splits list and return lists of each model parameters        
        1.4    type == "get_all", returns all parameters in list.        

-   Uncertainty_Func.py
    1. create_posterior(), generates posterior distribution of LPM by calibrating off historic data.
    2. plot_posterior2D(), returns plot of posterior distirbution found in 1.
    3. fit_mvn(), calculates optimum parameters of a multivariate normal distribution.
    4. construct_samples(), constructs samples from a multivariate normal distribution fitted to the data.
    5. plot_samples(), plots the samples generated from 4.
    6. model_ensemble(), solves the model for given parameter samples and plots using 5.    

-   PREVIOUS CODE FILES      
    1. Lab1Plot.py     
        - Generates plot of historical data (in 'Given?' part of report)  
    2. Model.py    
    3. Previous_Code.py    
