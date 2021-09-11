10/09/2021
#################### Resource Consent Project [Contamination in the Onehunga Aquifer] ####################
ENGSCI263 - Group 11
-   These functions are intended to aid our investigation into the effect of extraction from the Onehunga Aquifer 
    on the pressure and copper concentration, and hence help make an educated recommendation to the Auckland Regional 
    Council regarding the outcome of Watercare’s consent application.

##### Description #####
-   We created our LPM model of the aquifer pressure and copper concentration and optimised the unknown parameters by 
    minimising the sum of squares.
    
-   From this we then considered five different "what-if" scenarios:
    -   Stop operation in Onehunga Aquifer.
    -   Decrease extraction rate (5ML/day).
    -   Decrease extraction rate less (7.5ML/day).
    -   Continue operating at current rate (20ML/day).
    -   Double extraction rate (40 ML/day).

-   We then conduct uncertanty analysis by computing parameter posterior distribution of pressure and copper models (normal). 

-   This is used to fit to data and hence create samples for each senario. Then plot.

-   These plots allow us to visualise the scenarios and hence make a recommendation to Auckland Regional Council.

##### How to Use #####
-   First pull the "Resource-Management-Project" from the git repository.
-   Navigate to \Code and Data Files\main.py.
-   Read the header comments and decide which sections you would like to run.
    -   Switch section if statement to false if you do not wish to run.
-   Wait 20-40 seconds and inspect graph.

##### Data File Summary #####
-   ac_cu.csv, historic concentration measurements of Copper in aquifer (1985-2015, 5 year intervals, [mg/L]).
-   ac_p.csv, historic pressure measurements of pressure in aquifer (1985-2016, by-annual intervals, [MPa]).
-   ac_q.csv, historic extraction rates of aquifer operation (1985-2018, annual intervals, [10^6 L/day]).

##### Code File Summary (Python) #####
-   main.py
    1. Creates benchmarking plots.
    2. Calibrates model (must be completed for following sections)
    3. Generates plots of future senario predictions.
    4. Conducts uncertancy analysis.
    5. Generates plots of future senario predictions with uncertancy.

-   Copper.py
    1. ode_cu(), returns the derivative dC/dt at time, t, for given parameters.
    2. solve_ode_cu(), solves the Cu concentration ODE numerically (improved euler method).
    3. plot_aquifer_cu(), resolves the kettle LPM and plots over top of the data.
    4. evaluate_copper(), copper solution helper function for use in scipy.optimize.curve_fit.

-   Pressure.py
    1. ode_pressure(), returns the derivative dP/dt at time, t, for given parameters.
    2. solve_ode_pressure(), solves the pressure ODE numerically (improved euler method).
    3. plot_aquifer_pressure(), resolves the kettle LPM and plots over top of the data.
    4. evaluate_pressure(), pressure solution helper function for use in scipy.optimize.curve_fit.

-   test_solver.py
    1. .... ???? to be completed

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
    1. create_posterior(), generates posterior distribution of LPM by calibrating off hystoric data.
    2. plot_posterior2D(), plots posterior solution solved in 1.
    3. fit_mvn(), calculates optimum parameters of a multivariate normal distribution.
    4. construct_samples(), Constructs samples from a multivariate normal distribution fitted to the data.
    5. plot_samples(), Plots the samples generated from 4.
    6. model_ensemble(), Runs the model for given parameter samples and plots using 5.

-   PREVIOUS CODE FILES
    Lab1Plot.py
    Model.py
    Previuos_Code.py