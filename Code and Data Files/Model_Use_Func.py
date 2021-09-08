# Import libraries
import numpy as np
from Pressure import *
from Copper import *
from Helper_Func import *
#################################################################################################

### FUNCTIONS RELATED TO MODEL USE ###

# Solution Plotter Function (adds solution to the input axes, under the scenario inputted)
def plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_extraction, q, Parameters, historical, P, Cu, P_style, Cu_style, P_name, Cu_name, Cu_unit):

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

# Lumped Parameter Model Solver Function (solves the LPM in the data time domain, and evaluates at certain times, given the parameters)
def solve_lpm(t, Parameters):

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


def plot_aquifer_forecast_uncertainty(t0, t1, dt, P_ax, Cu_ax, t_extraction, q, Parameters, P, Cu, style):

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
        P_ax.plot(t_p_sol[:fut+1], P_sol[:fut+1], 'k', lw=0.3, alpha=0.2)
        P_ax.plot(t_p_sol[fut-1:], P_sol[fut-1:], style, lw=0.3, alpha=0.2)
    
    if Cu ==1:
        fut = np.where(t_cu_sol>2015)[0][0]
        Cu_ax.plot(t_cu_sol[:fut], Cu_sol[:fut], 'k', lw=0.3, alpha=0.2)
        Cu_ax.plot(t_cu_sol[fut-1:], Cu_sol[fut-1:], style, lw=0.3, alpha=0.2)

    return