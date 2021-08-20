# Import libraries
import numpy as np
from matplotlib import pyplot as plt


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


    ########## Generate best fit model plot, overlaying the pressure and copper models with the historical data ##########
    # Not all SI units:
    if True:   
        
        # Initialise model parameters 
        a = 1/(997*4184*0.0005) * 1.27          #Calibrate
        b = 1.5/(997*4184*0.0005) * 58          #Calibrate
        p0 = -5000                              #Calibrate, pressure at low pressure boundary (Pa)
        p1 = 5 * 10**4                          #Calibrate, pressure at high pressure boundary (Pa)
        p_init = 3.5*10**4
        
        C_src = 0.015                           #Calibrate, NOT SURE (g/m^3)
        c_init = 0.01                           #Calibrate, NOT SURE (g/m^3)????
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
    if True:   
        # Initialise model parameters 
        a = 1 * 10**-6                        #Calibrate
        b = 6 * 10**-2          #Calibrate
        p0 = 1000                              #Calibrate, pressure at low pressure boundary (Pa)
        p1 = 7 * 10**4                          #Calibrate, pressure at high pressure boundary (Pa)
        p_init = 3.5*10**4

        C_src = 8*10**-6                           #Calibrate
        c_init = 0                           
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