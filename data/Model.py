# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
#################################################################################################


### Pressure and copper derivative functions ###
def ode_pressure(t, P, q, a, b, p0, p1):
    ''' 
        Return the derivative dP/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable (time, in years)
        P : float
            Dependent variable (pressure, in Pa)
        q : float
            Mass flow rate of extraction rate (in kg/year)
        a : float
            Source/sink strength parameter for extraction
        b : float
            Recharge strength parameter for pressure recharge
        p0 : float
             Ambient pressure at low pressure boundary (in Pa)
        p1 : float
             Ambient pressure at high pressure boundary (in Pa)

        Returns:
        --------
        dPdt : float
               Derivative of aquifer pressure with respect to time

    '''

    # Evaluate derivative numerically at (t,P)
    dPdt = - a * q - b * (P - p0) - b * (P - p1)

    # Return derivative
    return dPdt

def ode_cu(t, P, C, a, b, d, p0, p1, C_src, M0):
    ''' 
        Return the derivative dC/dt at time, t, for given parameters.

        Parameters:
        -----------
        t :     float
                Independent variable (time, in years)
        P :     float
                Dependent variable (pressure, in Pa)
        C :     float
                Dependent variable (copper concentration, as mass fraction - unitless)
        a :     float
                Source/sink strength parameter for extraction
        b :     float
                Recharge strength parameter for pressure recharge
        d :     float
                Source/sink strength parameter for surface leaching
        p0 :    float
                Ambient pressure at low pressure boundary (in Pa)
        p1 :    float
                Ambient pressure at high pressure boundary (in Pa)
        Csrc :  float
                Copper concentration at surface source, as mass fraction (unitless)
       
        Returns:
        --------
        dCdt :  float
                Derivative of aquifer copper concentration with respect to time

    '''

    # Evaluate copper concentration at low pressure boundary, which depends on flow direction
    # Concentration same as C(t) if flow is out, otherwise take concentration as zero. 
    if P > p0:
        Cdash = C
    else:
        Cdash = 0

    # Evaluate derivative numerically at (t, P, C)
    dCdt = (-(b/a) * (P - p0) * (Cdash - C) + (b/a) * (P - p1) * C - d * (P - 0.5 * (p0 + p1))*C_src)/M0

    # Return derivative
    return dCdt


### Numerical solution functions for pressure and copper ###
def solve_ode_pressure(f, t0, t1, dt, t_data, q_data, P_parameters):
    ''' 
        Solves the pressure ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dP/dt given variable and parameter inputs.
        t0 : float
            Initial time of solution (year)
        t1 : float
            Final time of solution (year)
        dt : float
            Time step length (in years)
        p_init : float
            Initial pressure of the aquifer (in Pa)
        P_parameters : array-like
            List of parameters passed to ODE function f: a, b, p0, p1, p_init (in order)

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE should be solved using the Improved Euler Method. 

        Function q(t) should be hard coded within this method. Create duplicates of 
        solve_ode for models with different q(t).

        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters
    '''

    # Unpack parameters
    [a, b, p0, p1, p_init] = P_parameters

    # Calculate number of points to solve numerically
    npoints = int((t1 - t0) / dt + 1)

    # Initialise time and pressure solution vectors
    t = np.linspace(t0, t1, npoints)
    P = np.zeros(npoints)
    P[0] = p_init

    # Interpolate extraction rate at discrete solution points
    q = np.interp(t, t_data, q_data)

    # Iterate through solution points and solve numerically using Improved Euler method
    for i in range (0, npoints - 1):
        # Find euler estimate of next point
        edxdt = f(t[i], P[i], q[i], a, b, p0, p1) 
        ex1 = P[i] + dt*edxdt
        # Compute IE gradient
        iedxdt = f(t[i + 1], ex1, q[i], a, b, p0, p1)
        # Compute and store IE estimate of pressure
        P[i+1] = P[i] + dt*(edxdt + iedxdt)/2

    # Return time and pressure solution vectors
    return t, P

def solve_ode_cu(f, t0, t1, dt, t_sol, P_sol, C_parameters):
    ''' Solve an ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE should be solved using the Improved Euler Method. 

        Function q(t) should be hard coded within this method. Create duplicates of 
        solve_ode for models with different q(t).

        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters
    '''
    
    # Unpack parameters
    [a, b, d, p0, p1, c_init, C_src, M0] = C_parameters

    # Calculate number of points to solve numerically
    npoints = int((t1 - t0) / dt + 1)

    # Initialise time and copper concentration solution vectors
    t = np.linspace(t0, t1, npoints)
    C = np.zeros(npoints)
    C[0] = c_init
    # Obtain pressure values at discrete solution points, using inputted pressure solution
    p = np.interp(t, t_sol, P_sol)
    
    # Iterate through solution points and solve numerically using Improved Euler method
    for i in range (0, npoints - 1):
        # Find euler estimate of next point
        edxdt = f(t[i], p[i], C[i], a, b, d, p0, p1, C_src, M0)
        ex1 = C[i] + dt*edxdt
        # Compute IE gradient
        iedxdt = f(t[i + 1], p[i+1], ex1, a, b, d, p0, p1, C_src, M0)
        # Compute and store IE estimate of copper concentration
        C[i+1] = C[i] + 0.5 * dt * (edxdt + iedxdt)

    # Return time and copper concentration solution vectors
    return t, C


### Plotting functions for pressyre and copper ###
def plot_aquifer_pressure(t0, t1, dt, t_q_data, q_data, t_p_data, p_data, P_parameters):
    ''' Plot the kettle LPM over top of the data.

        Parameters:
        -----------
        none

        Returns:
        --------
        none

        Notes:
        ------
        This function called within if __name__ == "__main__":

        It should contain commands to read and plot the experimental data, run and 
        plot the kettle LPM for hard coded parameters, and then either display the 
        plot to the screen or save it to the disk.

    '''

    # Obtain the model solution
    m_time, m_press = solve_ode_pressure(ode_pressure, t0, t1, dt, t_q_data, q_data, P_parameters)

    # Plot the observations and model solutions
    plt.scatter(t_p_data, p_data, c = 'b', label="Observations")
    plt.plot(m_time, m_press, c = 'r', label="Model Solution")

    # Configure and show plot
    plt.legend()
    plt.ylabel('Pressure (Pa)')
    plt.xlabel('Time (Year)')
    plt.title('Best Fit Pressure Model')
    plt.show()

    # Return model solutions of time and pressure, to use as inputs for plot_aquifer_cu()
    return m_time, m_press

def plot_aquifer_cu(t0, t1, dt, t_sol, P_sol, t_cu_data, cu_data, C_parameters):
    ''' Plot the kettle LPM over top of the data.

        Parameters:
        -----------
        none

        Returns:
        --------
        none

        Notes:
        ------
        This function called within if __name__ == "__main__":

        It should contain commands to read and plot the experimental data, run and 
        plot the kettle LPM for hard coded parameters, and then either display the 
        plot to the screen or save it to the disk.

    '''
    
    # Obtain the model solution
    m_time, m_cu = solve_ode_cu(ode_cu, t0, t1, dt, t_sol, P_sol, C_parameters)


    # Plot the observations and model solutions
    plt.scatter(t_cu_data, cu_data, color = 'b', label="Observations")
    plt.plot(m_time, m_cu, c = 'r', label="Model Solution")

    # Configure and show plot
    plt.legend()
    plt.ylabel('Copper Concentration (mg/L OR mass fraction)')
    plt.xlabel('Time (Year)')
    plt.title('Best Fit Copper Concentration Model')
    plt.show()

    # Return model solutions of time and copper concentration
    return m_time, m_cu


#################################################################################################


### Gradient descent functions ###

def objective(theta):
    '''
    COMPLETE DOCSTRING
    theta = parameters = [a,b,p0,p1,p_init,c_init,C_src,d,M0,rho_sol]
    Returns S(theta)
    Note: pressure and copper observations are hard coded in
    '''

    # Unpack parameters from input theta
    [a, b, p0, p1, p_init, c_init, C_src, d, M0, rho_sol] = theta
    theta = [a,b, c_init,C_src]
    P_parameters = [a, b, p0, p1, p_init]
    C_parameters = [a, b, d, p0, p1, c_init, C_src, M0]

    # Read in extraction, pressure and copper concentration data
    data1 = np.genfromtxt("ac_q.csv", dtype=float, skip_header=1, delimiter=', ')
    t_q_data = data1[:,0]    
    q_data = data1[:,1] * 10**6 * 365                         # Extraction
    data2 = np.genfromtxt("ac_cu.csv", dtype=float, skip_header=1, delimiter=', ')
    t_cu_data = data2[:,0]
    cu_data = np.array(data2[:,1]) * 10**-3 / rho_sol         # Copper conc
    data3 = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ')
    t_p_data = data3[:,0]
    p_data = np.array(data3[:,1]) * 10**6                     # Pressure

    # Initialise time conditons to use the model for
    t0 = 1980; t1 = 2016; dt = 0.5

    # Evaluate model estimates at pressure points
    t_sol1, p1 = solve_ode_pressure(ode_pressure, t0, t1, dt, t_q_data, q_data, P_parameters)
    p_sol = np.interp(t_p_data, t_sol1, p1)
    p_sol = np.array(p_sol)

    # Evaluate model estimates at copper concentration points
    t_sol2, cu2 = solve_ode_cu(ode_cu, t0, t1, dt, t_sol1, p1, C_parameters)
    Cu_sol = np.interp(t_cu_data, t_sol2, cu2)
    Cu_sol = np.array(Cu_sol)

    # Evaluate the pressure misfit
    p_misfit = sum(((p_data - p_sol)/10**6)**2)

    # Evaluate the copper concentration misfit
    cu_misfit = sum(((cu_data - Cu_sol)*10**6)**2)

    # Evaluate a weighted sum
    S = p_misfit + cu_misfit

    return S

def objective_dir(objective, theta):
    """ 
        Compute a unit vector of objective function sensitivities, dS/dtheta.

        Parameters
        ----------
        obj: callable
            Objective function.
        theta: array-like
            Parameter vector at which dS/dtheta is evaluated.
        
        Returns
        -------
        s : array-like
            Unit vector of objective function derivatives.

    """
    # Empty list to store components of objective function derivative 
    s = np.zeros(len(theta))
    
    # Compute objective function at theta
    s0 = objective(theta)

    # Amount by which to increment parameter
    dtheta = 10**-3
    
    # For each parameter
    for i in range(len(theta)):
        # Basis vector in parameter direction 
        eps_i = np.zeros(len(theta))
        eps_i[i] = 1.
        
        # Compute objective function at incremented parameter
        si = objective(theta + dtheta * eps_i)

        # Compute objective function sensitivity
        s[i] = (si - s0)/dtheta

    # Normalise sensitivity vector
    s = s/np.linalg.norm(s)

    # Return sensitivity vector
    return s

def step(theta0, s, alpha):
    """ 
        Compute parameter update by taking step in steepest descent direction.

        Parameters
        ----------
        theta0 : array-like
            Current parameter vector.
        s : array-like
            Step direction.
        alpha : float
            Step size.
        
        Returns
        -------
        theta1 : array-like
            Updated parameter vector.
    """
    # Compute new parameter vector as sum of old vector and steepest descent step
    theta1 = theta0 - alpha * s
    
    return theta1

def line_search(objective, theta, s):
    """ Compute step length that minimizes objective function along the search direction.

        Parameters
        ----------
        obj : callable
            Objective function.
        theta : array-like
            Parameter vector at start of line search.
        s : array-like
            Search direction (objective function sensitivity vector).
    
        Returns
        -------
        alpha : float
            Step length.
    """
    # initial step size
    alpha = 0.
    # objective function at start of line search
    s0 = objective(theta)
    # anonymous function: evaluate objective function along line, parameter is a
    sa = lambda a: objective(theta-a*s)
    # compute initial Jacobian: is objective function increasing along search direction?
    j = (sa(.01)-s0)/0.01
    # iteration control
    N_max = 500
    N_it = 0
    # begin search
        # exit when (i) Jacobian very small (optimium step size found), or (ii) max iterations exceeded
    while abs(j) > 1.e-5 and N_it<N_max:
        # increment step size by Jacobian
        alpha += -j
        # compute new objective function
        si = sa(alpha)
        # compute new Jacobian
        j = (sa(alpha+0.01)-si)/0.01
        # increment
        N_it += 1
    # return step size
    return alpha

def gradient_descent(theta0):
    
    theta_all = [theta0]
    s0 = objective_dir(objective, theta0)
    s_all = [s0]

    # iteration control
    N_max = 500
    N_it = 0

    # begin steepest descent iterations
        # exit when max iterations exceeded
    while N_it < N_max:
        # uncomment line below to implement line search (TASK FIVE)
        alpha = line_search(objective, theta_all[-1], s_all[-1])
        
        # Update parameter vector 
        theta_next = step(theta0, s0, alpha)
        theta_all.append(theta_next) 	# save parameter value for plotting
        
        # Compute new direction for line search
        s_next = objective_dir(objective, theta_all[-1])
        s_all.append(s_next) 			# save search direction for plotting
        
        # Compute magnitude of steepest descent direction for exit criteria
        N_it += 1
        # Restart next iteration with values at end of previous iteration
        theta0 = 1.*theta_next
        s0 = 1.*s_next

    return theta0


#################################################################################################


if __name__ == "__main__":
    
    # The following code generates all the relevant plots and figures

    ########## Generate benchmark plots for both the pressure and copper numerical solvers ##########
    if False:
        # *** 1. Benchmarking for solve_ode_pressure ***
        # We will use the simplified condition that the extraction is constant, q(t) = q0
        # The analytical solution to dP/dt = -aq0 -b(P-p0) - b(P-p1) is, using the integrating factor method, 
        # P(t) = (-aq0/2b + (p0 + p1)/2) + e^-2bt (P_init + aq0/2b - (p0 + p1)/2)
        # Computing the analytical and numerical solutions for a simple set of parameters and conditions,
        t0 = 0; t1 = 4000; dt = 50; q0 = 20000; p0 = 1*10**5; p1 = 1.2*10**5; a = 0.001; b = 6*10**-4; p_init = 9*10**4
        P_parms = [a,b, p0, p1, p_init]
        npoints = int((t1 - t0) / dt + 1)
        times = np.linspace(t0, t1, npoints)
        P_analytical = (-a*q0/(2*b) + (p0 + p1)/2) + (np.e**(-2*b*times))*(p_init + a*q0/(2*b) - (p0 + p1)/2)
        # Initialise created data for numerical case:
        t_data = [t0, t1]; q_data = [q0, q0]
        _, P_numerical = solve_ode_pressure(ode_pressure, t0, t1, dt, t_data, q_data, P_parms)
        # Plot the solutions 
        f,host = plt.subplots(1,2)
        num, = host[0].plot(times, P_numerical, 'bx')
        ana, = host[0].plot(times, P_analytical, 'r-')
        #host[0].set_ylim([23000, 32000])
        host[0].set_title('Benchmark Plot for Pressure ODE Solver [Simple Parameters]')
        host[0].set_xlabel('Time, t (Year)')
        host[0].set_ylabel('Pressure, P (Pa)')
        host[0].legend([num, ana],['Numerical Solution', 'Analytical Solution'])
        
        # 2. *** Benchmarking for solve_ode_cu ***
        # We will use the simplified condition that the pressure is constant, P(t) = P < p0 at all times [so C' = 0]
        # The analytical solution to M0 dC/dt = -b/a(P-p0)(C' - C) + b/a(P-p1)C - d(P- (p0+p1)/2)C_src is, using the integrating factor method, 
        # C(t) = -k1/k2 + (c_init + k1/k2)e^k2t, where k1 = (-bC'(P-p0)/a - dC_scr(P - (p0+p1)/2))/M0 and k2 = b(2P - p0 - p1)/(aM0)
        # Computing the analytical and numerical solutions for a simple set of parameters and conditions,
        t0 = 0; t1 = 10; dt = 0.1; a = 4.2; b = 2.7; P = 1.4; c_init = 3; d = 1; C_src = 0.1; M0 = 1.2; Cdash = 0; p0=1.6; p1=1.9
        npoints = int((t1 - t0) / dt + 1)
        times = np.linspace(t0, t1, npoints)
        C_parms = [a, b, d, p0, p1, c_init, C_src, M0]
        k1 = (-b*Cdash*(P-p0)/a - d*C_src*(P - (p0+p1)/2))/M0
        k2 = b*(2*P - p0 - p1)/(a*M0)
        C_analytical = -k1/k2 + (c_init + k1/k2)*np.e**(k2*times)
        # Initialise created data for numerical case:
        t_sol = [0, t1]; P_sol = [P, P]
        _, C_numerical = solve_ode_cu(ode_cu, t0, t1, dt, t_sol, P_sol, C_parms)
        # Plot the solutions
        num, = host[1].plot(times, C_numerical, 'bx')
        ana, = host[1].plot(times, C_analytical, 'r-')
        host[1].set_title('Benchmark Plot for Copper ODE Solver [Simple Parameters]')
        host[1].set_xlabel('Time, t (Year)')
        host[1].set_ylabel('Copper Concentratiom (mg/L)')
        host[1].legend([num, ana],['Numerical Solution', 'Analytical Solution'])
        
        # Show plot
        plt.show()


    ########## Generate ROUGH/EYE-FIT model plots, overlaying the pressure and copper models with the historical data ##########
    # Not all SI units:
    if False:   
        
        # Initialise model parameters 
        a = 1/(997*4184*0.0005) * 1.27          #Calibrate
        b = 1.5/(997*4184*0.0005) * 58          #Calibrate
        p0 = -5000                              #Calibrate, pressure at low pressure boundary (Pa)
        p1 = 5 * 10**4                          #Calibrate, pressure at high pressure boundary (Pa)
        p_init = 3.5*10**4
        
        c_init = 0.01
        C_src = 0.015                           #Calibrate, NOT SURE (g/m^3)                           #Calibrate, NOT SURE (g/m^3)????
        d = 7500                                #Calibrate, NOT SURE ??????
        M0 = 8*10**7
        rho_w = 1000 
        
        # Read in extraction, copper concentration and pressure data from files
        # Extraction
        data1 = np.genfromtxt("ac_q.csv", dtype=float, skip_header=1, delimiter=', ')
        q_data = data1[:,1] * 10**6 * 0.365 #m^3/year
        t_q_data = data1[:,0]
        # Copper concentration
        data2 = np.genfromtxt("ac_cu.csv", dtype=float, skip_header=1, delimiter=', ')
        t_cu_data = data2[:,0]
        cu_data = data2[:,1]
        # Pressure
        data3 = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ')
        t_p_data = data3[:,0]
        p_data = data3[:,1] * 10**6     # in Pa

        # Create parameter vectors
        P_parameters = [a, b, p0, p1, p_init]
        C_parameters = [a, b, d, p0, p1, c_init, C_src, M0]

        # Specify solution domain
        t0 = 1980                               #Year start
        t1 = 2016                               #Year end
        dt = 1                          

        # fig, ax = plt.subplots(1,2)
        t_sol, P_sol = plot_aquifer_pressure(t0, t1, dt, t_q_data, q_data, t_p_data, p_data, P_parameters)
        plot_aquifer_cu(t0, t1, dt, t_sol, P_sol, t_cu_data, cu_data, C_parameters)
    # SI units for copper concentration and extraction rate:
    if False:   

        # Initialise model parameters 
        a = 1 * 10**-6                        #Calibrate
        b = 6 * 10**-2          #Calibrate
        p0 = 1000                              #Calibrate, pressure at low pressure boundary (Pa)
        p1 = 7 * 10**4                          #Calibrate, pressure at high pressure boundary (Pa)
        p_init = 3.5*10**4

        c_init = 0 
        C_src = 8*10**-6                           #Calibrate                          
        d = 12500                                #Calibrate
        M0 = 1*10**11
        rho_sol = 1000 
        
        # Read in extraction, copper concentration and pressure data from files
        # Extraction
        data1 = np.genfromtxt("ac_q.csv", dtype=float, skip_header=1, delimiter=', ')
        q_data = data1[:,1] * 10**6 * 365               # Convert from 10^6 L/day to kg/year
        t_q_data = data1[:,0]
        # Copper concentration
        data2 = np.genfromtxt("ac_cu.csv", dtype=float, skip_header=1, delimiter=', ')
        t_cu_data = data2[:,0]
        cu_data = data2[:,1] * 10**-3 / rho_sol         # Convert from mg/L to mass fraction
        # Pressure
        data3 = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ')
        t_p_data = data3[:,0]
        p_data = data3[:,1] * 10**6                     # Convert from MPa to Pa

        # Create parameter vectors
        P_parameters = [a, b, p0, p1, p_init]
        C_parameters = [a, b, d, p0, p1, c_init, C_src, M0]

        # Specify solution domain
        t0 = 1980                               #Year start
        t1 = 2016                               #Year end
        dt = 1                          


        # fig, ax = plt.subplots(1,2)
        t_sol, P_sol = plot_aquifer_pressure(t0, t1, dt, t_q_data, q_data, t_p_data, p_data, P_parameters)
        plot_aquifer_cu(t0, t1, dt, t_sol, P_sol, t_cu_data, cu_data, C_parameters)


    ########## Generate best fit model plots, using gradient descent ##########
    if False:

        # We will minimise the combined misfit of the pressure and copper models. In order to weight them approximately evenly,
        # it is necessary to scale the copper misfit up (since the copper cooncentration scale is many magnitudes less than the pressure scale)

        # Initialise model parameter estimates
        a = 1 * 10**-6                        
        b = 6 * 10**-2         
        p0 = 1000                              
        p1 = 7 * 10**4                          
        p_init = 3.5*10**4
        c_init = 0 
        C_src = 8*10**-6                                                    
        d = 12500
        M0 = 1*10**11
        rho_sol = 1000 
       
        # Read in extraction, pressure and copper concentration data
        data1 = np.genfromtxt("ac_q.csv", dtype=float, skip_header=1, delimiter=', ')
        t_q_data = data1[:,0]    
        q_data = data1[:,1] * 10**6 * 365                         # Extraction
        data2 = np.genfromtxt("ac_cu.csv", dtype=float, skip_header=1, delimiter=', ')
        t_cu_data = data2[:,0]
        cu_data = np.array(data2[:,1]) * 10**-3 / rho_sol         # Copper conc
        data3 = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ')
        t_p_data = data3[:,0]
        p_data = np.array(data3[:,1]) * 10**6                     # Pressure

        # Initialise initial parameter list estimate
        theta0 = [a,b,p0,p1,p_init,c_init,C_src,d,M0,rho_sol]
        P_parameters = [a, b, p0, p1, p_init]
        C_parameters = [a, b, d, p0, p1, c_init, C_src, M0]

        # Compute optimal theta
        theta_opt = gradient_descent(theta0)
        [a,b,p0,p1,p_init,c_init,C_src,d,M0,rho_sol] = theta_opt
        P_parameters = [a, b, p0, p1, p_init]
        C_parameters = [a, b, d, p0, p1, c_init, C_src, M0]


        # Plot optimal model
        t0 = 1980; t1 = 2016; dt = 0.5

        plot_aquifer_pressure(t0, t1, dt, t_q_data, q_data, t_p_data, p_data, P_parameters)
        print(*theta_opt)


    ########## Generate best fit model plots, using curve fitting ##########
    if True:

        # Estimate paramater values
        a = 1 * 10**-6                        
        b = 6 * 10**-2         
        p0 = 1000                              
        p1 = 7 * 10**4                          
        p_init = 3.5*10**4
        c_init = 0 
        C_src = 8*10**-6                                                    
        d = 12500
        M0 = 1*10**11
        rho_sol = 1000 

        # Read in extraction, pressure and copper concentration data from files
        data1 = np.genfromtxt("ac_q.csv", dtype=float, skip_header=1, delimiter=', ')
        t_q_data = data1[:,0]    
        q_data = data1[:,1] * 10**6 * 365                         # Extraction
        data2 = np.genfromtxt("ac_cu.csv", dtype=float, skip_header=1, delimiter=', ')
        t_cu_data = data2[:,0]
        cu_data = np.array(data2[:,1]) * 10**-3 / rho_sol         # Copper conc
        data3 = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ')
        t_p_data = data3[:,0]
        p_data = np.array(data3[:,1]) * 10**6                     # Pressure


        P_parameters = [a, b, p0, p1, p_init]
        C_parameters = [a, b, d, p0, p1, c_init, C_src, M0]


        def fit_ode_pressure(t, a, b, p0, p1, p_init):

            # Initialise data and time domain
            t0 = 1980; t1 = 2016; dt= 0.5
            data = np.genfromtxt("ac_q.csv", dtype=float, skip_header=1, delimiter=', ')
            t_data = data[:,0]    
            q_data = data[:,1] * 10**6 * 365                         

            # Calculate number of points to solve numerically
            npoints = int((t1 - t0) / dt + 1)

            # Initialise time and pressure solution vectors
            tsol = np.linspace(t0, t1, npoints)
            P = np.zeros(npoints)
            P[0] = p_init

            # Interpolate extraction rate at discrete solution points
            q = np.interp(tsol, t_data, q_data)

            # Iterate through solution points and solve numerically using Improved Euler method
            for i in range (0, npoints - 1):
                # Find euler estimate of next point
                edxdt = ode_pressure(tsol[i], P[i], q[i], a, b, p0, p1) 
                ex1 = P[i] + dt*edxdt
                # Compute IE gradient
                iedxdt = ode_pressure(tsol[i + 1], ex1, q[i], a, b, p0, p1)
                # Compute and store IE estimate of pressure
                P[i+1] = P[i] + dt*(edxdt + iedxdt)/2

            # Return pressure solution
            P_sol = np.interp(t, tsol,P )
            return P_sol
        theta_opt = curve_fit(fit_ode_pressure, t_p_data, p_data, p0 = P_parameters)[0]
        P_parameters = theta_opt
        [t_sol1, P_sol] = plot_aquifer_pressure(1980, 2016, 0.5, t_q_data, q_data, t_p_data, p_data, P_parameters)


        # Store pressure results to file
        fp1 = open('Intermediate_P_Solution.csv', 'w')
        fp1.write('Time_Sol, P_Sol')
        for i in range(len(t_sol1)):
            string1 = '\n{}, {}'.format(t_sol1[i], P_sol[i])
            fp1.write(string1)
        fp1.close()
        
        # Store parameter values to file
        #fp2 = open('Intermediate_P_parameters.csv', 'w')
        #fp2.write('P_parameters\n')
        #string2 = '{}, {}, {}, {}, {}'.format(*P_parameters)
        #fp2.write(string2)
        #fp2.close()


        def fit_ode_cu(t, a, b, d, p0, p1, c_init, C_src, M0):
            
            # Read in pressure solution data
            data = np.genfromtxt("Intermediate_P_solution.csv", dtype=float, skip_header=1, delimiter=', ')
            t_sol = data[:,0]    
            p_sol = data[:,1]

            # Read in parameters already obtained
            # data = np.genfromtxt("Intermediate_P_parameters.csv", dtype=float, skip_header=1, delimiter=', ')
            # [a, b, p0, p1, p_init] = data

            # Initialise data and time domain
            t0 = 1980; t1 = 2016; dt= 0.5
            # Calculate number of points to solve numerically
            npoints = int((t1 - t0) / dt + 1)

            # Initialise time and copper concentration solution vectors
            tsol = np.linspace(t0, t1, npoints)
            C = np.zeros(npoints)
            C[0] = c_init
            # Obtain pressure values at discrete solution points, using inputted pressure solution
            p = np.interp(tsol, t_sol, P_sol)
            
            # Iterate through solution points and solve numerically using Improved Euler method
            for i in range (0, npoints - 1):
                # Find euler estimate of next point
                edxdt = ode_cu(tsol[i], p[i], C[i], a, b, d, p0, p1, C_src, M0)
                ex1 = C[i] + dt*edxdt
                # Compute IE gradient
                iedxdt = ode_cu(tsol[i + 1], p[i+1], ex1, a, b, d, p0, p1, C_src, M0)
                # Compute and store IE estimate of copper concentration
                C[i+1] = C[i] + 0.5 * dt * (edxdt + iedxdt)

            # Return copper concentration solution
            C_val = np.interp(t, tsol, C)
            return C_val
        theta_opt = curve_fit(fit_ode_cu, t_cu_data, cu_data, p0 = C_parameters)[0]
        C_parameters = theta_opt
        [t_sol2, Cu_sol] = plot_aquifer_cu(1980, 2016, 0.5, t_sol1, P_sol, t_cu_data, cu_data, C_parameters)
        print(C_parameters)

        