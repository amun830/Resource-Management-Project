########################################## Pressure.py ##########################################
    # 1. ode_pressure(), returns the derivative dP/dt at time, t, for given parameters.
    # 2. solve_ode_pressure(), solves the pressure ODE numerically (improved euler method).
    # 3. plot_aquifer_pressure(), resolves the kettle LPM and plots over top of the data.
    # 4. evaluate_pressure(), pressure solution helper function for use in scipy.optimize.curve_fit.


# Import libraries
import numpy as np
from matplotlib import pyplot as plt


#################################################################################################

# 1. 

# Pressure ODE Evaluator Function
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
    #rate = extraction - recharge at low pressure boundary - recharge at high pressure boundary
    dPdt = - a * q - b * (P - p0) - b * (P - p1)

    # Return derivative
    return dPdt


#################################################################################################

# 2.

# Pressure ODE Solver Function
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
        t_data : array-like
            [t0, t1]
        q_data : array-like
            [Initial extraction rate, Final extraction rate]
        P_parameters : array-like
            List of parameters passed to ODE function f: a, b, p0, p1, p_init (in order)

        Returns:
        --------
        t : array-like
            Time variable solution vector (years).
        P : array-like
            Pressure variable solution vector (Pa).

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
        iedxdt = f(t[i + 1], ex1, q[i + 1], a, b, p0, p1)
        # Compute and store IE estimate of pressure
        P[i+1] = P[i] + dt*(edxdt + iedxdt)/2

    # Return time and pressure solution vectors
    return t, P


#################################################################################################

# 3.

# Pressure ODE Solver and Grapher Function
def plot_aquifer_pressure(t0, t1, dt, t_q_data, q_data, t_p_data, p_data, P_parameters):
    ''' 
        Plot the kettle LPM over top of the data.

        Parameters:
        -----------
        t0 : float
            Initial time of solution (year)
        t1 : float
            Final time of solution (year)
        dt : float
            Time step length (in years)
        t_q_data : array-like
            List of time points of given extraction data (q) 
        q_data : array-like
            List of extraction rates from given data.
        t_p_data : array-like
            List of time points of given pressure data. 
        p_data : array-like
            List of pressure values from given data.
        P_parameters : array-like
            List of parameters passed to ODE function f: a, b, p0, p1, p_init (in order)

        Returns:
        --------
        m_time : array-like
            List of time points of pressure LPM data.
        m_press : array-like
            List of pressure values of pressure LPM data.

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


#################################################################################################

# 4.

# Pressure Solution Evaluator Function (to pass into curve_fit during model calibration)
def evaluate_pressure(f, t0, t1, dt, t_q_data, q_data):
    ''' 
        Pressure solution helper function for use in scipy.optimize.curve_fit.

        Parameters:
        -----------
        t0 : float
            Initial time of solution (year)
        t1 : float
            Final time of solution (year)
        dt : float
            Time step length (in years)
        t_sol : array-like
            List of time points of given extraction data (q) 
        q_sol : array-like
            List of extraction rates from given data

        Returns:
        --------
        fit_pressure : callable
            Function to allow scipy.optimize.curve_fit() to callibrate optimum parameter values for pressure part of LPM.
    '''

    def fit_pressure(t, *P_parameters):
        t_sol, P_sol = solve_ode_pressure(f, t0, t1, dt, t_q_data, q_data, P_parameters)
        return np.interp(t, t_sol, P_sol)

    return fit_pressure


