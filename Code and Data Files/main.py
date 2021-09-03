# Import libraries and functions
import numpy as np
from matplotlib import pyplot as plt
from Pressure import *
from Copper import *
#################################################################################################

########## Generate benchmark plots for both the pressure and copper numerical solvers ##########
if True:
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
	# Show timestep convergence plot
	f.set_size_inches(15,9)
	plt.tight_layout(pad=1.5, h_pad=2.5, w_pad=2.5, rect=None)
	plt.savefig("Benchmarking_Plots.png")
	plt.show()






