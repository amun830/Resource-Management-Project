# Import libraries
import numpy as np
from Pressure import *
from Copper import *
from Helper_Func import *
#################################################################################################

### FUNCTIONS RELATED TO MODEL USE ###

# Solution Plotter Function (adds solution to input plots, under the scenario inputted)
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


