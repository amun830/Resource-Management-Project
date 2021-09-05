# Import libraries and functions
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from Pressure import *
from Copper import *
from Model_Use_Func import *
from Helper_Func import *

# Enter main function
if __name__ == "__main__":

	#################################################################################################

	########## Generate benchmark plots for both the pressure and copper numerical solvers ##########
	if False:
		# *** 1. Benchmarking for solve_ode_pressure ***
		# We will use the simplified condition that the extraction is constant, q(t) = q0
		# The analytical solution to dP/dt = -aq0 -b(P-p0) - b(P-p1) is, using the integrating factor method, 
		# P(t) = (-aq0/2b + (p0 + p1)/2) + e^-2bt (P_init + aq0/2b - (p0 + p1)/2)
		# Computing the analytical and numerical solutions for a simple set of parameters and conditions,
		t0 = 0; t1 = 25; dt = 1; q0 = 1*10**8; p0 = 1000; p1 = 4*10**4; a = 1*10**-5; b = 9*10**-2; p_init = 5*10**2; t_data = [t0, t1]; q_data = [q0, q0]
		P_parms = [a,b, p0, p1, p_init]
		P_times, P_numerical = solve_ode_pressure(ode_pressure, t0, t1, dt, t_data, q_data, P_parms)
		P_analytical = (-a*q0/(2*b) + (p0 + p1)/2) + (np.e**(-2*b*P_times))*(p_init + a*q0/(2*b) - (p0 + p1)/2)
		# Plot the solutions 
		f,(host, host2, host3) = plt.subplots(3,2)
		num, = host[0].plot(P_times, P_numerical, 'bx')
		ana, = host[0].plot(P_times, P_analytical, 'r-')
		host[0].set_title('Benchmark Plot for Pressure ODE Solver [Simple Parameters]')
		host[0].set_xlabel('Time, t (Year)')
		host[0].set_ylabel('Pressure, P (Pa)')
		host[0].legend([num, ana],['Numerical Solution', 'Analytical Solution'])
		
		# 2. *** Benchmarking for solve_ode_cu ***
		# We will use the simplified condition that the pressure is constant, P(t) = P < p0 at all times [so C' = 0]
		# The analytical solution to M0 dC/dt = -b/a(P-p0)(C' - C) + b/a(P-p1)C - d(P- (p0+p1)/2)C_src is, using the integrating factor method, 
		# C(t) = -k1/k2 + (c_init + k1/k2)e^k2t, where k1 = (-bC'(P-p0)/a - dC_scr(P - (p0+p1)/2))/M0 and k2 = b(2P - p0 - p1)/(aM0)
		# Computing the analytical and numerical solutions for a simple set of parameters and conditions,
		t0 = 0; t1 = 10; dt = 0.25; a = 4.2; b = 2.7; P = 1.4; c_init = 3; d = 1; C_src = 0.1; M0 = 1.2; Cdash = 0; p0=1.6; p1=1.9; t_sol = [0, t1]; P_sol = [P, P]
		C_parms = [a, b, d, p0, p1, c_init, C_src, M0]
		k1 = (-b*Cdash*(P-p0)/a - d*C_src*(P - (p0+p1)/2))/M0
		k2 = b*(2*P - p0 - p1)/(a*M0)
		C_times, C_numerical = solve_ode_cu(ode_cu, t0, t1, dt, t_sol, P_sol, C_parms)
		C_analytical = -k1/k2 + (c_init + k1/k2)*np.e**(k2*C_times)
		# Plot the solutions
		num, = host[1].plot(C_times, C_numerical, 'bx')
		ana, = host[1].plot(C_times, C_analytical, 'r-')
		host[1].set_title('Benchmark Plot for Copper ODE Solver [Simple Parameters]')
		host[1].set_xlabel('Time, t (Year)')
		host[1].set_ylabel('Copper Conc., C (mg/L)')
		host[1].legend([num, ana],['Numerical Solution', 'Analytical Solution'])

		# *** 3. Error analysis plots ***
		# Compute percentage error in the numerical solutions for both pressure and copper concentration (obtained in 1. and 2.)
		P_error = 100*(P_numerical - P_analytical)/P_analytical
		C_error = 100*(C_numerical - C_analytical)/C_analytical
		# Plot the errors
		host2[0].plot(P_times, P_error, 'k'); host2[1].plot(C_times, C_error, 'k'); 
		host2[0].set_title('Error Analysis of Pressure Solution'); host2[1].set_title('Error Analysis of Copper Conc. Solution')
		host2[0].set_xlabel('Time, t (Year)'); host2[1].set_xlabel('Time, t (Year)')
		host2[0].set_ylabel('% Error (against benchmark)'); host2[1].set_ylabel('% Error (against benchmark)')
		
		# *** 4. Timestep convergence plots ***
		# Compute the numerical solution at t = t1 using different time steps dt, for both ODE solver functions
		# a. Pressure:
		# Initialise parameters
		t0 = 0; t1 = 25; q0 = 1*10**8; p0 = 1000; p1 = 4*10**4; a = 1*10**-5; b = 9*10**-2; p_init = 5*10**2; t_data = [t0, t1]; q_data = [q0, q0]
		P_parms = [a,b, p0, p1, p_init]
		# Initialise solution array
		P_sol = np.zeros(100)
		P_dt_values = np.linspace(0.1,3,100)
		P_dt_recip = 1/P_dt_values
		for i in range(100): # Loop through different dt
			dt = P_dt_values[i]
			P_sol[i] = solve_ode_pressure(ode_pressure, t0, t1, dt, t_data, q_data, P_parms)[1][-1]
		# b. Copper Concentration:
		# Initialise parameters
		t0 = 0; t1 = 10; a = 4.2; b = 2.7; P = 1.4; c_init = 3; d = 1; C_src = 0.1; M0 = 1.2; Cdash = 0; p0=1.6; p1=1.9; t_sol_sim = [0, t1]; P_sol_sim = [P, P]
		C_parms = [a, b, d, p0, p1, c_init, C_src, M0]
		# Initialise solution array
		C_sol = np.zeros(70)
		C_dt_values = np.linspace(0.1,2,70)
		C_dt_recip = 1/C_dt_values
		for i in range(70): # Loop through different dt
			dt = C_dt_values[i]
			C_sol[i] = solve_ode_cu(ode_cu, t0, t1, dt, t_sol_sim, P_sol_sim, C_parms)[1][-1]
		# Plot the solutions
		host3[0].plot(P_dt_recip, P_sol, 'k.'); host3[1].plot(C_dt_recip, C_sol, 'k.')
		host3[0].set_title('Timestep Convergence of Pressure Solution'); host3[1].set_title('Timestep Convergence of Copper Conc. Solution')
		host3[0].set_xlabel('Reciprocal of timestep, 1/dt (1/Year)'); host3[1].set_xlabel('Reciprocal of timestep, 1/dt (1/Year)')
		host3[0].set_ylabel('P(t = 25)'); host3[1].set_ylabel('C(t = 10)')
		# Configure timestep convergence plot
		f.set_size_inches(15,9)
		plt.tight_layout(pad=1.5, h_pad=2.5, w_pad=2.5, rect=None)
		plt.savefig("Benchmarking_Plots.png")
		# plt.show()


	#################################################################################################

	########## 			Calibrate model to the historical data (all SI units)			   ##########
	if True:   # Note this condition must be set to TRUE for all of the code following it to work

		# Initialise estimates of parameter values
		a = 1 * 10**-6                       
		b = 6 * 10**-2          
		p0 = 1000                              
		p1 = 7 * 10**4                          
		p_init = 3.5*10**4
		d = 12500
		c_init = 0 
		C_src = 8*10**-6                                                    
		M0 = 1*10**11
		rho_sol = 1000 		# Note this parameter is assumed to be rho_water at 25°C, as the water extracted is very dilute
		# Store parameter estimates in the respective vectors
		P_parameters = [a, b, p0, p1, p_init]
		Extra_C_parameters = [d, c_init, C_src, M0]


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


		# Plot both the calibrated pressure and copper concentration models against the historical data, as a visual check
		# f, P_ax = plt.subplots(figsize=(14,6))
		# Cu_ax = P_ax.twinx()
		# plt.title("Calibrated Model Against Historical Data"); P_ax.set_xlabel("Year"); P_ax.set_ylabel("Aquifer Pressure (MPa)"); Cu_ax.set_ylabel("Copper Concentration (mg/L)")
		# p,cu, p_hist, cu_hist = plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_q_data, q_data, theta_all, historical=True, P=1, Cu=1, P_style="k", Cu_style="r", P_name = "Pressure (Model)", Cu_name = "Copper Conc. (Model)", Cu_unit = "mg/L")
		# P_ax.legend(handles=[p, cu, p_hist, cu_hist], loc = 4)


	#################################################################################################

	##########	        Use model to simulate "What-if" scenarios and forecast     	       ##########
	if True:
		
		### SCENARIO MODELLING ###
		
		# Period to model into the future:
		predict = 60 
		t1 += predict
		# Initialise and configure plots: set the following to True to have separate plots
		combined = False
		if combined == True:
			f, P_ax = plt.subplots(figsize=(14,8))
			Cu_ax = P_ax.twinx()		
			plt.title("Model Solution for Pressure and Copper Concentration of the Onehunga Aquifer")
			P_ax.set_xlabel("Year")
			P_ax.set_ylabel("Aquifer Pressure (MPa)"); Cu_ax.set_ylabel("Copper Concentration (mg/L)")
		else:
			f_P, P_ax = plt.subplots(figsize=(12,8)); f_Cu, Cu_ax = plt.subplots(figsize=(12,8))
			P_ax.set_xlabel("Year"); Cu_ax.set_xlabel("Year")
			P_ax.set_ylabel("Aquifer Pressure (MPa)"); Cu_ax.set_ylabel("Copper Concentration (mg/L)")
			P_ax.set_title("Scenario Modelling for the Onehunga Aquifer Pressure"); Cu_ax.set_title("Scenario Modelling for the Onehunga Aquifer Copper Concentration")

		# Initialise loop lists
		scenarios = [40, 20, 7.5, 5, 0] 			# Different extraction scenarios to model, in 10^6 L/day
		styles = ['m', 'g', 'b', 'c', 'pink']		# Corresponding plot styles
		P_handles = []; Cu_handles = []

		# Plot the different scenarios
		for i in range(len(scenarios)):
			outcome = scenarios[i]
			style = styles[i]
			t_q_data_future = np.concatenate([t_q_data, [t_q_data[-1]+0.00001, t_q_data[-1]+predict]])
			q_data_future = np.concatenate([q_data, 2*[outcome * 10**3 * 365 * rho_sol]])
			name = "{} ML/day".format(outcome)
			p, cu, _ , _ = plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_q_data_future, q_data_future, theta_all, True, 1, 1, style, style, name, name, "mg/L")
			P_handles.append(p)
			Cu_handles.append(cu)

		# Add a line for the recommended copper concentration limits
		# A safety factor of 1.5 is applied, to the health limit of 2mg/L (Maximum Allowable Value for Health as stated in the Drinking Water Standards for NZ 2008)
		h_limit = 2/1.5
		Cu_ax.hlines(y=h_limit, xmin=t0, xmax=t1, color='slategrey', linestyle='--')
		Cu_ax.annotate("Health Limit (with 1.5 Safety Factor)", (1980, 1.35), color="slategrey", size="9")
		# The aesthetic guideline of 1mg/L (as stated in the Drinking Water Standards for NZ 2008)
		a_limit = 1
		Cu_ax.hlines(y=a_limit, xmin=t0, xmax=t1, color='slategrey', linestyle='--')
		Cu_ax.annotate("Guideline Aesthetic Determinand", (1980, 1.02), color="slategrey", size="9")

		# Add legend based on whether the data is combined into one plot (or not)
		if combined == True:
			P_ax.legend(handles=P_handles, loc = 0)

		else:
			P_ax.legend(handles=P_handles, loc = 0); Cu_ax.legend(handles=Cu_handles, loc = 4)
			
			# Add annotations to the plots
			Cu_ax.annotate("Double", (2060, 1.42))
			Cu_ax.annotate("No change", (2060, 1.26))
			Cu_ax.annotate("Reduced", (2060, 1.1))
			Cu_ax.annotate("Reduced more", (2060, 1.02))
			Cu_ax.annotate("Stop", (2060, 0.84))
			P_ax.annotate("Double", (2063, -0.081))
			P_ax.annotate("No change", (2063, -0.023))
			P_ax.annotate("Reduced", (2063, 0.005))
			P_ax.annotate("Reduced more", (2063, 0.02))
			P_ax.annotate("Stop", (2063, 0.034))


		# Show plot
		plt.show()


	#################################################################################################

	##########         			    Conduct Uncertainty Analysis	  			   	       ##########



