########## main.py ##########
    # 1. Creates benchmarking plots.
    # 2. Calibrates model (must be completed for following sections)
    # 3. Generates plots of future senario predictions.
    # 4. Conducts uncertancy analysis.
    # 5. Generates plots of future senario predictions with uncertancy.

    # Note - Toggle if statements to 'True' to run desired sections.


# Import libraries and functions
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size
from scipy.optimize import curve_fit
from Pressure import *
from Copper import *
from Model_Use_Func import *
from Helper_Func import *
from Uncertainty_Func import *

# Enter main function
if __name__ == "__main__":

    #################################################################################################

    # 1.

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
        # We will use the simplified condition that the pressure is constant, P(t) = P > p0 at all times [so C' = 0]
        # The analytical solution to M0 dC/dt = -b/a(P-p0)(C' - C) + b/a(P-p1)C - d(P- (p0+p1)/2)C_src is, using the integrating factor method, 
        # C(t) = -k1/k2 + (c_init + k1/k2)e^k2t, where k1 = (-bC'(P-p0)/a - dC_scr(P - (p0+p1)/2))/M0 and k2 = b(2P - p0 - p1)/(aM0)
        # Computing the analytical and numerical solutions for a simple set of parameters and conditions,
        t0 = 0; t1 = 10; dt = 0.25; a = 4.2; b = 2.7; P = 1.4; c_init = 3; dC_src = 0.1; M0 = 1.2; Cdash = 0; p0=1.6; p1=1.9; t_sol = [0, t1]; P_sol = [P, P]
        C_parms = [a, b, dC_src, p0, p1, c_init, M0]
        k1 = (-b*Cdash*(P-p0)/a - dC_src*(P - (p0+p1)/2))/M0
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
        t0 = 0; t1 = 10; a = 4.2; b = 2.7; P = 1.4; c_init = 3; dC_src = 0.1; M0 = 1.2; Cdash = 0; p0=1.6; p1=1.9; t_sol_sim = [0, t1]; P_sol_sim = [P, P]
        C_parms = [a, b, dC_src, p0, p1, c_init, M0]
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


    #################################################################################################

    # 2a. 
    
    ########## 		Calibrate model to the historical data (all SI units) - Attempt 1	   ##########
    if True:    # Note this condition must be set to True for all of the code following it to work
        
        # Initialise estimates of parameter values
        a = 1 * 10**-6                       
        b = 6 * 10**-2          
        p0 = 1000                              
        p1 = 7 * 10**4                       
        p_init = 3.5*10**4
        dC_src = 12500 * 8*10**-6
        c_init = 0 
        M0 = 1*10**11
        rho_sol = 1000 	

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

        all_time_data = np.append(t_p_data, t_cu_data)
        all_data = np.append(p_data, cu_data)

        # Define function for curve_fit
        def combined_fit(t0, t1, dt, t_q_data, q_data, t_p_data, t_cu_data):

            def combinedFunction(all_times, *Parameters):
                
                # Single data reference passed in, extract separate data
                p_times = all_times[:len(t_p_data)] # first data
                cu_times = all_times[len(t_p_data):] # second data

                P_parameters, Cu_parameters = get_parameter_set(Parameters, None, "split")

                t_sol, P_sol = solve_ode_pressure(ode_pressure, t0, t1, dt, t_q_data, q_data, P_parameters)
                result1 = np.interp(p_times, t_sol, P_sol)

                t_sol2, Cu_sol = solve_ode_cu(ode_cu, t0, t1, dt, t_sol, P_sol, Cu_parameters)
                result2 = np.interp(cu_times, t_sol2, Cu_sol)

                return np.append(result1, result2)

            return combinedFunction

        # Fit model
        all_mean, all_var = curve_fit(combined_fit(t0, t1, dt, t_q_data, q_data, t_p_data, t_cu_data), all_time_data, all_data, [a, b, p0, p1, p_init,dC_src, c_init, M0], bounds=(0,[1,1, 10**5,10**7,10**7, 1000, 1, np.inf]))

        # Plot calibrated model
        f, P_ax = plt.subplots(figsize=(14,6))
        Cu_ax = P_ax.twinx()
        plt.title("Calibrated Model Against Historical Data"); P_ax.set_xlabel("Time (Year)"); P_ax.set_ylabel("Aquifer Pressure (MPa)"); Cu_ax.set_ylabel("Copper Concentration (mg/L)")
        p,cu, p_hist, cu_hist = plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_q_data, q_data, all_mean, historical=True, P=1, Cu=1, P_style="k", Cu_style="r", P_name = "Pressure (Model)", Cu_name = "Copper Conc. (Model)", Cu_unit = "mg/L")
        P_ax.legend(handles=[p, cu, p_hist, cu_hist], loc = 4)
        print("First calibration best-fit parameters: {}".format(all_mean))

        # Plot misfit
        f_mis, ax_mis = plot_model_misfit(all_mean, t_p_data, 10**-6 * p_data, t_cu_data, 10**6 * cu_data)


    #################################################################################################

    # 2b.

    ########## 		Calibrate model to the historical data (all SI units) - Attempt 2	   ##########
    if True:   # Note this condition must be set to True for all of the code following it to work

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
       
        # Calibrate pressure model parameters
        theta_P, cov_P = curve_fit(evaluate_pressure(ode_pressure, t0, t1, dt, t_q_data, q_data), t_p_data, p_data, p0 = P_parameters, bounds=(0,[1,10,10**5,10**7,10**5]))
        # Solve pressure model using this set of calibrated parameters
        t_sol, P_sol = solve_ode_pressure(ode_pressure, t0, t1, dt, t_q_data, q_data, theta_P)
        # Calibrate copper concentration model parameters, using the above pressure solution
        theta_C_extra, cov_C_extra = curve_fit(evaluate_copper(ode_cu, t0, t1, dt, t_sol, P_sol, theta_P), t_cu_data, cu_data, p0 = Extra_C_parameters)  # Could add bounds=([-np.inf, -1,0],[np.inf, 1, np.inf])
        # Combine parameters into the calibrated parameter vector, _all_all
        theta_all = get_parameter_set(theta_P, theta_C_extra, "theta_all")

        # Find variance of parameter estimates
        var_P = np.diag(cov_P); var_C_extra = np.diag(cov_C_extra)
        var_all = np.concatenate([var_P,var_C_extra])

        # Plot both the calibrated pressure and copper concentration models against the historical data, as a visual check
        f, P_ax = plt.subplots(figsize=(14,6))
        Cu_ax = P_ax.twinx()
        plt.title("Calibrated Model Against Historical Data"); P_ax.set_xlabel("Time (Year)"); P_ax.set_ylabel("Aquifer Pressure (MPa)"); Cu_ax.set_ylabel("Copper Concentration (mg/L)")
        p,cu, p_hist, cu_hist = plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_q_data, q_data, theta_all, historical=True, P=1, Cu=1, P_style="k", Cu_style="r", P_name = "Pressure (Model)", Cu_name = "Copper Conc. (Model)", Cu_unit = "mg/L")
        P_ax.legend(handles=[p, cu, p_hist, cu_hist], loc = 4)

    #################################################################################################

    # 3.

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
            P_ax.set_xlabel("Time (Year)")
            P_ax.set_ylabel("Aquifer Pressure (MPa)"); Cu_ax.set_ylabel("Copper Concentration (mg/L)")
        else:
            f_P, P_ax = plt.subplots(figsize=(12,8)); f_Cu, Cu_ax = plt.subplots(figsize=(12,8))
            P_ax.set_xlabel("Time (Year)"); Cu_ax.set_xlabel("Time (Year)")
            P_ax.set_ylabel("Aquifer Pressure (MPa)"); Cu_ax.set_ylabel("Copper Concentration (mg/L)")
            P_ax.set_title("Scenario Modelling for the Onehunga Aquifer Pressure"); Cu_ax.set_title("Scenario Modelling for the Onehunga Aquifer Copper Concentration")

        # Initialise loop lists
        scenarios = [40, 20, 10, 5, 0] 			# Different extraction scenarios to model, in 10^6 L/day
        styles = ['r', 'lime', 'b', 'c', 'm']		# Corresponding plot styles
        P_handles = []; Cu_handles = []

        # Plot the different scenarios
        for i in range(len(scenarios)):
            outcome = scenarios[i]
            style = styles[i]
            t_q_data_future = np.concatenate([t_q_data, [t_q_data[-1]+0.00001, t_q_data[-1]+predict]])
            q_data_future = np.concatenate([q_data, 2*[outcome * 10**3 * 365 * rho_sol]])
            name = "{} ML/day".format(outcome)
            p, cu, _ , _ = plot_aquifer_model(t0, t1, dt, P_ax, Cu_ax, t_q_data_future, q_data_future, theta_all, False, 1, 1, style, style, name, name, "mg/L")
            P_handles.append(p)
            Cu_handles.append(cu)
        
        # Plot the historical data
        p, cu, p_hist, cu_hist = plot_aquifer_model(t0, t1-predict, dt, P_ax, Cu_ax, t_q_data, q_data, theta_all, True, 1, 1, 'k-', 'k-', "Best fit model", "Best fit model", "mg/L")
        P_handles.append(p); P_handles.append(p_hist)
        Cu_handles.append(cu); Cu_handles.append(cu_hist)

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
            Cu_ax.annotate("Double", (2060, 1.45))
            Cu_ax.annotate("No change", (2060, 1.28))
            Cu_ax.annotate("Reduced", (2060, 1.18))
            Cu_ax.annotate("Reduced more", (2060, 1.04))
            Cu_ax.annotate("Stop", (2060, 0.864))
            P_ax.annotate("Double", (2063, -0.083))
            P_ax.annotate("No change", (2063, -0.023))
            P_ax.annotate("Reduced", (2063, 0.007))
            P_ax.annotate("Reduced more", (2063, 0.0213))
            P_ax.annotate("Stop", (2063, 0.036))


    #################################################################################################

    # 4.

    ##########         			    Conduct Uncertainty Analysis	  			   	       ##########
    if True:

        # Generate posterior distributions for the parameter set [a, b, dC_src, M0]
        # These have been done separately for visual aid, but combined when generating parameter sets
        
        # Pressure parameter posterior distribution
        a,b,P1 = create_posterior(theta_all, 80, quantity="pressure")
        # plot_posterior2D(a,b,"a (1/ms^2)", "b (1/s)", "Posterior of 2 Pressure Parameters", P1)
        # Copper parameter posterior distribution
        dC_src,M0,P2 = create_posterior(theta_all, 80, quantity="copper")
        # plot_posterior2D(dC_src,M0,"dC_src (ms)", "M0 (kg)", "Posterior of 2 Copper Conc. Parameters", P2)

        n_samples = 100
        ab_samples = construct_samples(a,b, theta_all, P1, n_samples, "pressure", False)
        # model_ensemble(ab_samples, theta_all, "pressure")
        dCM0_samples = construct_samples(dC_src,M0, theta_all, P2, n_samples, "copper", False)
        # model_ensemble(dCM0_samples, theta_all, "copper")


    #################################################################################################

    # 5.

    ##########	        Simulate "What-if" scenarios and forecast WITH UNCERTAINTY         ##########
    if True:
        
        ### SCENARIO MODELLING ###
        
        # Period to model into the future:
        predict2 = 60
        t1 = 2016 + predict2
        # Initialise and configure plots: set the following to True to have separate plots
        combined = False
        if combined == True:
            f, P_ax = plt.subplots(figsize=(14,8))
            Cu_ax = P_ax.twinx()		
            plt.title("Model Solution for Pressure and Copper Concentration of the Onehunga Aquifer")
            P_ax.set_xlabel("Time (Year)")
            P_ax.set_ylabel("Aquifer Pressure (MPa)"); Cu_ax.set_ylabel("Copper Concentration (mg/L)")
        else:
            f_P, P_ax = plt.subplots(figsize=(12,8)); f_Cu, Cu_ax = plt.subplots(figsize=(12,8))
            P_ax.set_xlabel("Time (Year)"); Cu_ax.set_xlabel("Time (Year)")
            P_ax.set_ylabel("Aquifer Pressure (MPa)"); Cu_ax.set_ylabel("Copper Concentration (mg/L)")
            P_ax.set_title("Scenario Modelling for the Onehunga Aquifer Pressure"); Cu_ax.set_title("Scenario Modelling for the Onehunga Aquifer Copper Concentration")

        # Initialise loop lists
        scenarios = [40, 20, 10, 5, 0] 			# Different extraction scenarios to model, in 10^6 L/day
        styles = ['r', 'lime', 'b', 'c', 'm']		# Corresponding plot styles
        Param = np.copy(theta_all)
        coppermax_times = np.array([])
        coppermax_val = np.array([])
        pressure_steady = np.array([])

        # Plot the different scenarios
        for i in range(len(scenarios)):
            outcome = scenarios[i]
            style = styles[i]
            t_q_data_future = np.concatenate([t_q_data, [t_q_data[-1]+0.00001, t_q_data[-1]+predict2]])
            q_data_future = np.concatenate([q_data, 2*[outcome * 10**3 * 365 * rho_sol]])
            name = "{} ML/day".format(outcome)
            
            case_interval_pressure = np.array([])
            case_interval_copper = np.array([])

            for j in range(len(ab_samples)):
                Param[5] = dCM0_samples[j][0]
                Param[7] = dCM0_samples[j][1]
                P_sol, _, _ = plot_aquifer_forecast_uncertainty(t0, t1, dt, P_ax, Cu_ax, t_q_data_future, q_data_future, Param, 0, 1, style)
                
                Param[0:2] = ab_samples[j]
                _, t_cu_sol, Cu_sol = plot_aquifer_forecast_uncertainty(t0, t1, dt, P_ax, Cu_ax, t_q_data_future, q_data_future, Param, 1, 0, style)

                # Store interval prediction data if extraction is the recommended amount
                if outcome == 5:
                    index = np.argmax(Cu_sol)
                    coppermax_val = np.append(coppermax_val, Cu_sol[index])
                    coppermax_times = np.append(coppermax_times, t_cu_sol[index])
                    pressure_steady = np.append(pressure_steady,P_sol[-1])
                
                case_interval_pressure = np.append(case_interval_pressure, P_sol[-1])
                case_interval_copper = np.append(case_interval_copper, Cu_sol[-1])

            # Add scenario labels
            P_ax.plot([],[], style,lw=1.2, alpha=1, label=name)
            Cu_ax.plot([],[], style, lw=1.2, alpha=1, label=name)

            # Compute prediction intervals (long-term value of property)
            case_interval_pressure_90 = [np.percentile(case_interval_pressure,5), np.percentile(case_interval_pressure,95)]
            case_interval_copper_90 = [np.percentile(case_interval_copper,5), np.percentile(case_interval_copper,95)]
            print("90% pressure and copper interval after 60 years for {}: {}, {}".format(name, case_interval_pressure_90, case_interval_copper_90))

        # Add best fit model labels        
        P_ax.plot([],[], 'k-',lw=1.2, alpha=1, label="Best fit models")
        Cu_ax.plot([],[], 'k-',lw=1.2, alpha=1, label="Best fit models")


        # Add a line for the recommended copper concentration limits
        # A safety factor of 1.5 is applied, to the health limit of 2mg/L (Maximum Allowable Value for Health as stated in the Drinking Water Standards for NZ 2008)
        h_limit = 2/1.5
        Cu_ax.hlines(y=h_limit, xmin=t0, xmax=t1, color='slategrey', linestyle='--')
        Cu_ax.annotate("Health Limit (with 1.5 Safety Factor)", (1980, 1.35), color="slategrey", size="9")
        # The aesthetic guideline of 1mg/L (as stated in the Drinking Water Standards for NZ 2008)
        a_limit = 1
        Cu_ax.hlines(y=a_limit, xmin=t0, xmax=t1, color='slategrey', linestyle='--')
        Cu_ax.annotate("Guideline Aesthetic Determinand", (1980, 1.02), color="slategrey", size="9")

        # Plot historical data
        P_ax.errorbar(t_p_data,10**-6 * p_data,yerr=6*10**-3,fmt='ro', label=' Historical data', zorder= 1000)
        Cu_ax.errorbar(t_cu_data,10**6 * cu_data,yerr=5*10**-2,fmt='ro', label='Historical data', zorder= 1000)

        # Add legends
        P_ax.legend(loc=0)
        Cu_ax.legend(loc=4)

        # Compute 90% confidence interval for the time and value of the copper peak, under the recommendation
        time_int_90 = [np.percentile(coppermax_times,5), np.percentile(coppermax_times,95)]
        coppermax_int_90 = [np.percentile(coppermax_val,5), np.percentile(coppermax_val,95)]
        print("A 90% confidence interval for the peak copper concentration in the next 60 years, under the recommended maximum extraction of 5ML/day, is {}, in mg/L".format(coppermax_int_90))
        print("Also, the 90% confidence interval for the corresponding year that this occurs is {}".format(time_int_90))

        # Compute 90% confidence interval for the steasy state pressure under the recommendation
        pressure_steady_int_90 = [np.percentile(pressure_steady,5), np.percentile(pressure_steady,95)]
        print("With 5ML/day extraction, a 90% confidence interval for the long-term (steady) pressure is {}, in MPa".format(pressure_steady_int_90))


    #################################################################################################

    # Show all figures
    plt.show()