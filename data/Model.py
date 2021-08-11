# imports
import numpy as np
from matplotlib import pyplot as plt


def ode_pressure(t, P, q, a, b, p0, p1):
    ''' Return the derivative dx/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        P : float
            Dependent variable.
        q : float
            Source/sink rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        p0 : float
            Ambient value of dependent variable.

        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        >>> ode_model(0, 1, 2, 3, 4, 5)
        22

    '''

    dxdt = - a * q - b * (P - p0) - b * (P - p1)


    return dxdt

def ode_cu(t, P, C, Csrc, d, a, b, p0, p1):
    ''' Return the derivative dx/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        P : float
            Dependent variable.
        q : float
            Source/sink rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        p0 : float
            Ambient value of dependent variable.

        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        >>> ode_model(0, 1, 2, 3, 4, 5)
        22

    '''
    Cdash = (P > p0)*C

    dcdt = -(b/a) * (P - p0) * (Cdash - C) + (b/a) * (P - p0) * C - d * (P - (p0 -p1)/2)*Csrc


    return dcdt

def solve_ode_pressure(f, t0, t1, dt, p0, p1, pars):
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

    data = np.genfromtxt("ac_q.csv", dtype=float, skip_header=1, delimiter=', ')
    tv = data[:,0]
    qv = data[:,1] * 10**6 * 0.365 #m^3/year


    npoints = int((t1 - t0) / dt + 1)

    t = np.linspace(t0, t1, npoints)
    P = np.zeros(npoints)

    q = np.interp(t, tv, qv)

    P[0] = p0

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

def plot_aquifer_pressure(t0, t1, dt, p0, p1, a, b):
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

    timean, pressan = solve_ode_pressure(ode_pressure, t0, t1, dt, p0, p1, [a, b])
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

    a = 1/(997*4184*0.0005) #Calibrate
    b = 1.5/(997*4184*0.0005) #Calibrate
    t0 = 1980                       #Year start
    t1 = 2016                       #Year end
    dt = 1                          
    p0 = 3.6725020e-02 * 10**6      #Calibrate, pressure at low pressure boundary (Pa)
    p1 = 3.6725020e-02 * 10**6      #Calibrate, pressure at high pressure boundary (Pa)
    Csrc = 1000                     #Calibrate, NOT SURE (g/m^3)
    c0 = 1000                       #Calibrate, NOT SURE (g/m^3)????
    d = 1000                        #Calibrate, NOT SURE ??????

    plot_aquifer_pressure(t0, t1, dt, p0, p1, a, b)
    plot_aquifer_cu(t0, t1, dt, Csrc, c0, d, p0, p1, a, b)

    