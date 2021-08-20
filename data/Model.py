# imports
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

def ode_cu(t, P, C, Csrc, d, a, b, p0, p1):
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
        Csrc :  float
                Copper concentration at surface source, as mass fraction (unitless)
        d :     float
                Source/sink strength parameter for surface leaching
        a :     float
                Source/sink strength parameter for extraction
        b :     float
                Recharge strength parameter for pressure recharge
        p0 :    float
                Ambient pressure at low pressure boundary (in Pa)
        p1 :    float
                Ambient pressure at high pressure boundary (in Pa)
        Returns:
        --------
        dCdt :  float
                Derivative of aquifer copper concentration with respect to time

    '''

    # Evaluate copper concentration at low pressure boundary, which depends on flow direction
    # Concentration same as C(t) if flow is out, otherwise take concentration as zero. 
    Cdash = (P > p0)*C

    # Evaluate derivative numerically at (t, P, C)
    dCdt = -(b/a) * (P - p0) * (Cdash - C) + (b/a) * (P - p1) * C - d * (P - 0.5 * (p0 + p1))*Csrc

    # Return derivative
    return dCdt

def solve_ode_pressure(f, t0, t1, dt, t_data, q_data, p0, p1, pars):
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

    # Calculate number of points to solve numerically
    npoints = int((t1 - t0) / dt + 1)

    # Initialise time and pressure solution vectors
    t = np.linspace(t0, t1, npoints)
    P = np.zeros(npoints)
    P[0] = pars[2]

    # Interpolate extraction rate at discrete solution points
    q = np.interp(t, t_data, q_data)

    for i in range (0, npoints - 1):
        edxdt = f(t[i], P[i], q[i], pars[0], pars[1], p0, p1) 
        ex1 = P[i] + dt*edxdt

        iedxdt = f(t[i + 1], ex1, q[i], pars[0], pars[1], p0, p1)

        P[i+1] = P[i] + dt*(edxdt + iedxdt)/2

    return t, P


def solve_ode_cu(f, t0, t1, dt, Csrc, c0, d, p0, p1, pars):
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

    data = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ')
    tv = data[:,0]
    pv = data[:,1]*10**6 #Pa


    npoints = int((t1 - t0) / dt + 1)

    t = np.linspace(t0, t1, npoints)
    C = np.zeros(npoints)

    p = np.interp(t, tv, pv)

    C[0] = c0

    for i in range (0, npoints - 1):
        edxdt = f(t[i], p[i], C[i], Csrc, d, pars[0], pars[1], p0, p1) 
        ex1 = C[i] + dt*edxdt

        iedxdt = f(t[i + 1], p[i+1], ex1, Csrc, d, pars[0], pars[1], p0, p1)

        C[i+1] = C[i] + dt*(edxdt + iedxdt)/2

    return t, C

def plot_aquifer_pressure(t0, t1, dt, t_data, q_data, p0, p1, a, b, p_init):
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

    timean, pressan = solve_ode_pressure(ode_pressure, t0, t1, dt, t_data, q_data, p0, p1, [a, b, p_init])
    data = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ')
    time = data[:,0]
    press = data[:,1]*10**6     #Pa

    press = np.interp(timean, time, press)

    plt.scatter(timean, press, c = 'b', label="Numerical")
    plt.plot(timean, pressan, c = 'r', label="Analytical")

    plt.legend()
    plt.ylabel('Pressure')
    plt.xlabel('Time')
    plt.title('Pressure over time')

    plt.show()





def plot_aquifer_cu(t0, t1, dt, Csrc, c0, d, p0, p1, a, b):
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

    time_an, cu_an = solve_ode_cu(ode_cu, t0, t1, dt, Csrc, c0, d, p0, p1, [a, b])

    data = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ')
    time = data[:,0]
    cu = data[:,1]      #g/m^3

    cu = np.interp(time_an, time, cu)

    plt.scatter(time_an, cu, c = 'b', label="Numerical")
    plt.plot(time_an, cu_an, c = 'r', label="Analytical")

    plt.legend()
    plt.ylabel('Cu')
    plt.xlabel('Time')
    plt.title('Cu concentration over time')

    plt.show()

if __name__ == "__main__":
    
    # The following code generates all the relevant plots and figures

    ########## Generate benchmark plots for both the pressure and copper numerical solvers ##########
    if True:
        # *** 1. Benchmarking for solve_ode_pressure ***
        # We will use the simplified condition that the extraction is constant, q(t) = q0
        # The analytical solution to dP/dt = -aq0 -b(P-p0) - b(P-p1) is, using the integrating factor method, 
        # P(t) = (-aq0/2b + (p0 + p1)/2) + e^-2bt (P_init + aq0/2b - (p0 + p1)/2)
        # Computing the analytical and numerical solutions over the time domain [0, 100] using q0 = 1000, p0 = 10^5, p1 = 1.2 x 10^5, 
        # a = 1, b = 2, and P_init = 7 x 10^4 (and dt = 0.5)
        t0 = 0; t1 = 4000; q0 = 20000; p0 = 1*10**5; p1 = 1.2*10**5; a = 0.001; b = 6*10**-4; p_init = 9*10**4; dt = 50; pars = [a,b, p_init]
        npoints = int((t1 - t0) / dt + 1)
        times = np.linspace(t0, t1, npoints)
        P_analytical = (-a*q0/(2*b) + (p0 + p1)/2) + (np.e**(-2*b*times))*(p_init + a*q0/(2*b) - (p0 + p1)/2)
        # Initialise created data for numerical case:
        t_data = [t0, t1]; q_data = [q0, q0]
        _, P_numerical = solve_ode_pressure(ode_pressure, t0, t1, dt, t_data, q_data, p0, p1, pars)
        # Plot the solutions 
        f,host = plt.subplots(1,2)
        num, = host[0].plot(times, P_numerical, 'bx')
        ana, = host[0].plot(times, P_analytical, 'r-')
        #host[0].set_ylim([23000, 32000])
        host[0].set_title('Benchmark Plot for Pressure ODE Solver [Simple Parameters]')
        host[0].set_xlabel('Time, t (year)')
        host[0].set_ylabel('Pressure, P (Pa)')
        host[0].legend([num, ana],['Numerical Solution', 'Analytical Solution'])
        
        # 2. *** Benchmarking for solve_ode_cu ***

        # Show plot
        plt.show()




    data = np.genfromtxt("ac_q.csv", dtype=float, skip_header=1, delimiter=', ')
    q_data = data[:,1] * 10**6 * 0.365 #m^3/year
    t_data = data[:,0]

    a = 1/(997*4184*0.0005)         #Calibrate
    b = 1.5/(997*4184*0.0005)       #Calibrate
    t0 = 1980                       #Year start
    t1 = 2016                       #Year end
    dt = 1                          
    p0 = 1 * 10**5      #Calibrate, pressure at low pressure boundary (Pa)
    p1 = 1.2 * 10**5      #Calibrate, pressure at high pressure boundary (Pa)
    p_init = 0.7 * 10**5
    Csrc = 0.0001                     #Calibrate, NOT SURE (g/m^3)
    c0 = 0.0001                       #Calibrate, NOT SURE (g/m^3)????
    d = 20                        #Calibrate, NOT SURE ??????

    plot_aquifer_pressure(t0, t1, dt, t_data, q_data, p0, p1, a, b, p_init)
    plot_aquifer_cu(t0, t1, dt, Csrc, c0, d, p0, p1, a, b)

    