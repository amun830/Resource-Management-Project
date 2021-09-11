# Import libraries
import numpy as np
from Pressure import *
from Copper import *
from Helper_Func import *
from Model_Use_Func import *
from scipy.optimize import curve_fit
	
#################################################################################################

### SCENARIO MODELLING ###
# Initialise estimates of parameter values
a = 1 * 10**-6                       
b = 6 * 10**-2          
p0 = 1000                              
p1 = 7 * 10**4                       
p_init = 3.5*10**4
dC_src = 12500 * 8*10**-6
c_init = 0 
M0 = 1*10**11
rho_sol = 1000 		# Note this parameter is assumed to be rho_water at 25°C, as the water extracted is very dilute
# Store parameter estimates in the respective vectors
P_parameters = [a, b, p0, p1, p_init]
Extra_C_parameters = [dC_src, c_init, M0]

# Obtain historical data on extraction, pressure and copper conc. from files and convert to SI units where necessary
# Extraction rate data
t_q_data, q_data = np.genfromtxt("ac_q.csv", dtype=float, skip_header=True, delimiter=', ').T
q_data *= 10**3 * 365 * rho_sol 	# Convert from 10^6 L/day to kg/year
# Pressure data
t_p_data, p_data = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ').T
p_data *= 10**6						# Convert from MPa to Pa
# Copper concentration data
t_cu_data, cu_data = np.genfromtxt("ac_cu.csv", dtype=float, skip_header=1, delimiter=', ').T
cu_data *= 10**-3 / rho_sol			# Convert from mg/L to mass fraction (unitless)

# Specify solution domain to use for calibration (ie historical data domain)
t0 = 1980          # Year start
t1 = 2016          # Year end
dt = 0.5           # Timestep              
# Calibrate pressure model parameters
theta_P = curve_fit(evaluate_pressure(ode_pressure, t0, t1, dt, t_q_data, q_data), t_p_data, p_data, p0 = P_parameters, bounds=(0,[np.inf,np.inf,10**5,10**7,10**5]))[0]
# Solve pressure model using this set of calibrated parameters
t_sol, P_sol = solve_ode_pressure(ode_pressure, t0, t1, dt, t_q_data, q_data, theta_P)
# Calibrate copper concentration model parameters, using the above pressure solution
theta_C_extra = curve_fit(evaluate_copper(ode_cu, t0, t1, dt, t_sol, P_sol, theta_P), t_cu_data, cu_data, p0 = Extra_C_parameters)[0]
# Combine parameters into the calibrated parameter vector, theta_all
theta_all = get_parameter_set(theta_P, theta_C_extra, "theta_all")

if False:
    # Period to model into the future:
    predict = 60 
    t1 += predict
    # Initialise and configure plots: set the following to True to have separate plots
    combined = False
    if combined == True:
        f, P_ax = plt.subplots()
        Cu_ax = P_ax.twinx()		
        plt.title("Model Solution for Pressure and Copper Concentration of the Onehunga Aquifer")
        P_ax.set_xlabel("Year")
        P_ax.set_ylabel("Aquifer Pressure (MPa)")
        Cu_ax.set_ylabel("Copper Concentration (mg/L)")
    else:
        f_P, P_ax = plt.subplots(); f_Cu, Cu_ax = plt.subplots()
        P_ax.set_xlabel("Year"); Cu_ax.set_xlabel("Year")
        P_ax.set_ylabel("Aquifer Pressure (MPa)"); Cu_ax.set_ylabel("Copper Concentration (mg/L)")
        P_ax.set_title("Scenario Modelling for the Onehunga Aquifer Pressure"); Cu_ax.set_title("Scenario Modelling for the Onehunga Aquifer Copper Concentration")

    # *** 1. What-if scario #1 - increase the maximum allowable extraction to 40 x 10^6 L/day ***
    # We will model the case when the extraction is this maximum value every day
    t_q_data_future1 = np.concatenate([t_q_data, [t_q_data[-1]+0.00001, t_q_data[-1]+predict]])
    q_data_future1 = np.concatenate([q_data, 2*[40 * 10**3 * 365 * rho_sol]])
    p1, cu1, _ , _ = plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_q_data_future1, q_data_future1, theta_all, False, 1, 1, "m", "m", "40 ML/day", "40 ML/day", "mg/L")

    # *** 2. What-if scario #2 - do not change the maximum allowable extraction of 20 x 10^6 L/day ***
    # We will model the case when the extraction is this maximum value every day
    t_q_data_future2 = np.concatenate([t_q_data, [t_q_data[-1]+0.00001, t_q_data[-1]+predict]])
    q_data_future2 = np.concatenate([q_data, 2*[20 * 10**3 * 365 * rho_sol]])
    p2, cu2, _ , _ = plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_q_data_future2, q_data_future2, theta_all, False, 1, 1, "g", "g", "20 ML/day", "20 ML/day", "mg/L")

    # *** 3. What-if scario #3 - impose an indefinite moratorium on the aquifer usage ***
    t_q_data_future3 = np.concatenate([t_q_data, [t_q_data[-1]+0.00001, t_q_data[-1]+predict]])
    q_data_future3 = np.concatenate([q_data, [0,0]])
    p3, cu3, _ , _ = plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_q_data_future3, q_data_future3, theta_all, False, 1, 1, "y", "y", "0 ML/day", "0 ML/day", "mg/L")

    # *** 4. What-if scario #4 - reduce the resource consent, to a recommended safe level ***
    # We will model the cases when the extraction is the maximum value every day
    # Case a: Maximum daily extraction of 5 x 10^6 L/day
    t_q_data_future4a = np.concatenate([t_q_data, [t_q_data[-1]+0.00001, t_q_data[-1]+predict]])
    q_data_future4a = np.concatenate([q_data, 2*[5 * 10**3 * 365 * rho_sol]])
    p4a, cu4a, _ , _ = plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_q_data_future4a, q_data_future4a, theta_all, False, 1, 1, "c", "c", "5 ML/day", "5 ML/day", "mg/L")
    # Case b: Maximum daily extraction of 10 x 10^6 L/day
    t_q_data_future4b = np.concatenate([t_q_data, [t_q_data[-1]+0.00001, t_q_data[-1]+predict]])
    q_data_future4b = np.concatenate([q_data, 2*[7.5 * 10**3 * 365 * rho_sol]])
    p4b, cu4b, _ , _ = plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_q_data_future4b, q_data_future4b, theta_all, False, 1, 1, "r", "k", "7.5 ML/day", "7.5 ML/day", "mg/L")

    if combined == False:
        # Add legends to separate plots
        P_ax.legend(handles=[p1, p2, p4b, p4a, p3], loc = 0)
        Cu_ax.legend(handles=[cu1, cu2, cu4b, cu4a, cu3], loc = 0)

    else:
        P_ax.legend(handles=[p1, p2, p4b, p4a, p3], loc = 0)

    plt.show()

#################################################################################################
		
### CALIBRATION ### 
# Plot both the calibrated pressure and copper concentration models against the historical data, as a visual check
if False:
    f, P_ax = plt.subplots(figsize=(14,6))
    Cu_ax = P_ax.twinx()
    plt.title("Calibrated Model Against Historical Data"); P_ax.set_xlabel("Year"); P_ax.set_ylabel("Aquifer Pressure (MPa)"); Cu_ax.set_ylabel("Copper Concentration (mg/L)")
    p,cu, p_hist, cu_hist = plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_q_data, q_data, theta_all, historical=True, P=1, Cu=1, P_style="k", Cu_style="r", P_name = "Pressure (Model)", Cu_name = "Copper Conc. (Model)", Cu_unit = "mg/L")
    P_ax.legend(handles=[p, cu, p_hist, cu_hist], loc = 4)
    plt.show()

#################################################################################################

### UNCERTAINTY ANALYSIS ###
def create_posterior_combined(Parameters_best, N):
    
    # Unpack best estimates of parameter values
    a_best, b_best, p0_best, p1_best, p_init_best, dC_src_best, c_init_best, M0_best = get_parameter_set(Parameters_best, None, "get_all")

    # Generate vectors of parameter values
    par1 = np.linspace(a_best/2,a_best*2, N)            # a
    par2 = np.linspace(b_best/2,b_best*2, N)            # b

	# Create grid of all parameter combinations
    A, B = np.meshgrid(par1, par2, indexing='ij')
	# Initialise matrix for objective function
    S_p = np.zeros(A.shape)
    S_cu = np.zeros(A.shape)

    # Read in data for calibration
    t_p_data, p_data = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ').T
    t_cu_data, cu_data = np.genfromtxt("ac_cu.csv", dtype=float, skip_header=1, delimiter=', ').T


    # Set the error variance, at 0.005MPa for pressure and 0.02mg/L
    p_var = 10**-4
    cu_var = 0.01

    # Loop through all parameter combinations, and compute the sum of squares objective function for each
    for i in range(len(par1)):
        for j in range(len(par2)):
                p_model, _ = solve_lpm(t_p_data,[par1[i],par2[j],p0_best, p1_best, p_init_best, dC_src_best, c_init_best, M0_best])
                _, cu_model = solve_lpm(t_cu_data,[par1[i],par2[j],p0_best, p1_best, p_init_best, dC_src_best, c_init_best, M0_best])
                S_p[i,j] = np.sum((p_data-p_model)**2)/p_var
                S_cu[i,j] = np.sum((cu_data-cu_model)**2)/cu_var
            

    # Compute the posterior
    Posterior_p = np.exp(-S_p/2.)
    Posterior_cu = np.exp(-S_cu/2.)

    # Normalise to a probability density function
    Volume_p = np.sum(Posterior_p) * (par1[1]-par1[0]) * (par2[1]-par2[0])
    Volume_cu = np.sum(Posterior_cu) * (par1[1]-par1[0]) * (par2[1]-par2[0])

    Posterior_p /= Volume_p
    Posterior_cu /= Volume_cu

    Posterior = 0.5*Posterior_cu + 0.5*Posterior_p

    return par1, par2, Posterior


    all_time_data = np.append(t_p_data, t_cu_data)
    all_data = np.append(10**-6 * p_data, cu_data)

#################################################################################################

### CALIBRATION ###

if True: 
    all_time_data = np.append(t_p_data, t_cu_data)
    all_data = np.append(p_data, cu_data)

    def combined_fit(t0, t1, dt, t_q_data, q_data, t_p_data, t_cu_data):

        def combinedFunction(all_times, *Parameters):
            
            # single data reference passed in, extract separate data
            p_times = all_times[:len(t_p_data)] # first data
            cu_times = all_times[len(t_p_data):] # second data

            P_parameters, Cu_parameters = get_parameter_set(Parameters, None, "split")

            t_sol, P_sol = solve_ode_pressure(ode_pressure, t0, t1, dt, t_q_data, q_data, P_parameters)
            result1 = np.interp(p_times, t_sol, P_sol)

            t_sol2, Cu_sol = solve_ode_cu(ode_cu, t0, t1, dt, t_sol, P_sol, Cu_parameters)
            result2 = np.interp(cu_times, t_sol2, Cu_sol)

            return np.append(result1, result2)

        return combinedFunction

    all_mean, all_var = curve_fit(combined_fit(t0, t1, dt, t_q_data, q_data, t_p_data, t_cu_data), all_time_data, all_data, [a, b, p0, p1, p_init,dC_src, c_init, M0])


    f, P_ax = plt.subplots(figsize=(14,6))
    Cu_ax = P_ax.twinx()
    plt.title("Calibrated Model Against Historical Data"); P_ax.set_xlabel("Year"); P_ax.set_ylabel("Aquifer Pressure (MPa)"); Cu_ax.set_ylabel("Copper Concentration (mg/L)")
    p,cu, p_hist, cu_hist = plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_q_data, q_data, all_mean, historical=True, P=1, Cu=1, P_style="k", Cu_style="r", P_name = "Pressure (Model)", Cu_name = "Copper Conc. (Model)", Cu_unit = "mg/L")
    P_ax.legend(handles=[p, cu, p_hist, cu_hist], loc = 4)
    plt.show()