######################################### Model_Use_Func.py #########################################
    #1. plot_aquifer_model(), Solution Plotter Function.
    #2. plot_model_misfit(). Plots the misfit of a model (against data)
    #3. solve_lpm(), Lumped Parameter Model Solver Function.
    #4. plot_aquifer_forecast_uncertainty(), Plots forcasted LPM with uncertainty.


# Import libraries
import numpy as np
from Pressure import *
from Copper import *
from Helper_Func import *
#################################################################################################

#1. 

# Solution Plotter Function (adds solution to the input axes, under the scenario inputted)
def plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_extraction, q, Parameters, historical, P, Cu, P_style, Cu_style, P_name, Cu_name, Cu_unit):
    ''' 
        Plot the kettle LPM over top of the data.

        Parameters:
        -----------
        t0 :    float
            Initial time of solution (year)
        t1 :    float
            Final time of solution (year)
        dt :    float
            Time step length (in years)
        P_ax :  plt.plot()
            Exisisting pressure plot.
        Cu_ax : plt.plot()
            Exisisting Cu concentration plot.
        t_extraction :  array-like
            List of time points of given extraction data (q) 
        q : array-like
            List of extraction rates from given data.
        Parameters :    array-like
            List of LPM parameters.
        historical :    boolean
            True if we want to plot hystorical data.
        P :     int
            1 if we want to plot/return Pressure, 0 if else.
        Cu :     int
            1 if we want to plot/return Cu concentration, 0 if else.
        P_style :   string
            denoter of pressure plot line / fill style.
        Cu_style :   string
            denoter of Cu concentration plot line / fill style.
        P_name : string
            Label of Pressure plot line.
        Cu_name : string
            Label of Cu concentration plot line.
        Cu_unit :   string
            Unit of Cu concentration.

        Returns:
        --------
        p : plt.plot()
            Plot of Pressure.
        cu : plt.plot()
            Plot of Cu concentration.
        p_hist : plt.plot()
            Plot of historical pressure data.
        cu_hist : plt.plot()
            Plot of histprical Cu concentration.
    '''

    # Obtain the pressure and copper model parameters
    P_parameters, C_parameters = get_parameter_set(Parameters, None, "split")

    # Solve model
    t_p_sol, P_sol = solve_ode_pressure(ode_pressure, t0, t1, dt, t_extraction, q, P_parameters)
    t_cu_sol, Cu_sol = solve_ode_cu(ode_cu, t0, t1, dt, t_p_sol, P_sol, C_parameters)
    # Convert pressure data from Pa to MPa
    P_sol /= 10**6
    # Adjust copper concentration units if necessary
    if Cu_unit == "mg/L":
        Cu_sol *= 10**6
    

    # Initialise returns
    p, cu, p_hist, cu_hist = None, None, None, None

    # Read in and plot historical data if necessary
    if historical == True:
        # Read in pressure data
        t_p_data, p_data = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ').T
        # Read in copper concentration data
        t_cu_data, cu_data = np.genfromtxt("ac_cu.csv", dtype=float, skip_header=1, delimiter=', ').T
        cu_data *= 10**-6       			# Convert from mg/L to mass fraction (unitless)

        # Adjust copper concentration units if necessary
        if Cu_unit == "mg/L":
            cu_data *= 10**6

        # Plot historical data
        if P == 1:
            p_hist, = P_ax.plot(t_p_data, p_data, 'k.', label="Pressure Data")
        if Cu == 1:
            cu_hist, = Cu_ax.plot(t_cu_data, cu_data,'r.', label="Copper Conc. Data")

    # Plot model
    if P == 1:
        p, = P_ax.plot(t_p_sol, P_sol, P_style, label=P_name)

    if Cu == 1:
        cu, = Cu_ax.plot(t_cu_sol, Cu_sol, Cu_style, label=Cu_name)
    
    return p, cu, p_hist, cu_hist


#################################################################################################

#2.

def plot_model_misfit(Parameters, t_p_data, p_data, t_cu_data, cu_data):

    # Solve model at times
    f, ax = plt.subplots(nrows=1,ncols=2,figsize=(14,5))
    P_model, _ = solve_lpm(t_p_data, Parameters)
    _, Cu_model = solve_lpm(t_cu_data, Parameters)

    # Compute misfit
    P_misfit = P_model - p_data
    Cu_misfit = Cu_model - cu_data

    # Configure plot
    ax[0].plot(t_p_data, P_misfit, 'kx')
    ax[0].set_title("Pressure Model Misfit")
    ax[0].set_xlabel("Time (Year)")
    ax[0].set_ylabel("Pressure Misfit (MPa)")
    ax[1].plot(t_cu_data, Cu_misfit, 'rx')
    ax[1].set_title("Copper Concentration Model Misfit")
    ax[1].set_xlabel("Time (Year)")
    ax[1].set_ylabel("Copper Conc. Misfit (mg/L)")
    ax[0].axhline(y=0, color= 'grey', linestyle='--')
    ax[1].axhline(y=0, color= 'grey', linestyle='--')

    return f, ax

#################################################################################################

#3. 


# Lumped Parameter Model Solver Function (solves the LPM in the data time domain, and evaluates at certain times, given the parameters)
def solve_lpm(t, Parameters):
    '''
        Parameters:
        ------------
        t : array-like
            Desired time values of LPM solution.
        Parameters :    array-like
            List of LPM parameters.
        
        Returns:
        ------------
        P_int : array-like
            Solved and interpolated Pressure solution to LPM.
        Cu_int :    array-like
            Solved and interpolated Cu concentration solution to LPM.
    '''

    # Read in extraction data, in kg/year
    t_q_data, q_data = np.genfromtxt("ac_q.csv", dtype=float, skip_header=True, delimiter=', ').T 
    q_data *= 10**6 * 365 

    # Obtain the pressure and copper model parameters
    P_parameters, C_parameters = get_parameter_set(Parameters, None, "split")

    # Solve model
    t_p_sol, P_sol = solve_ode_pressure(ode_pressure, 1980, 2018, 0.5, t_q_data, q_data, P_parameters)
    t_cu_sol, Cu_sol = solve_ode_cu(ode_cu, 1980, 2018, 0.5, t_p_sol, P_sol, C_parameters)

    # Convert units back to same as data collected (MPa and mg/L)
    P_sol /= 10**6
    Cu_sol *= 10**6

    # Interpolate model at inputted times
    P_int = np.interp(t, t_p_sol, P_sol)
    Cu_int = np.interp(t, t_cu_sol, Cu_sol)

    return P_int, Cu_int


#################################################################################################

#4. 

def plot_aquifer_forecast_uncertainty(t0, t1, dt, P_ax, Cu_ax, t_extraction, q, Parameters, P, Cu, style):
    ''' 
        Plot the forecasted LPM with uncertainty.

        Parameters:
        -----------
        t0 :    float
            Initial time of solution (year)
        t1 :    float
            Final time of solution (year)
        dt :    float
            Time step length (in years)
        P_ax :  plt.plot()
            Exisisting pressure plot.
        Cu_ax : plt.plot()
            Exisisting Cu concentration plot.
        t_extraction :  array-like
            List of time points of given extraction data (q) 
        q : array-like
            List of extraction rates from given data.
        Parameters :    array-like
            List of LPM parameters.
        P :     int
            1 if we want to plot/return Pressure, 0 if else.
        Cu :     int
            1 if we want to plot/return Cu concentration, 0 if else.
        style :   string
            Denoter of plot line / fill style

    '''

    # Obtain the pressure and copper model parameters
    P_parameters, C_parameters = get_parameter_set(Parameters, None, "split")

    # Solve model
    t_p_sol, P_sol = solve_ode_pressure(ode_pressure, t0, t1, dt, t_extraction, q, P_parameters)
    t_cu_sol, Cu_sol = solve_ode_cu(ode_cu, t0, t1, dt, t_p_sol, P_sol, C_parameters)
    # Convert pressure data from Pa to MPa
    P_sol /= 10**6
    # Adjust copper concentration units
    Cu_sol *= 10**6
    
    if P==1:
        fut = np.where(t_p_sol>2018)[0][0]
        P_ax.plot(t_p_sol[:fut], P_sol[:fut], 'k', lw=0.3, alpha=0.2)
        P_ax.plot(t_p_sol[fut-1:], P_sol[fut-1:], style, lw=0.3, alpha=0.2)
    
    if Cu ==1:
        fut = np.where(t_cu_sol>2015)[0][0]
        Cu_ax.plot(t_cu_sol[:fut], Cu_sol[:fut], 'k', lw=0.3, alpha=0.2)
        Cu_ax.plot(t_cu_sol[fut-1:], Cu_sol[fut-1:], style, lw=0.3, alpha=0.2)

    return P_sol, t_cu_sol, Cu_sol