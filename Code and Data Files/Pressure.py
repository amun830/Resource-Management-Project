# Import libraries
import numpy as np
from matplotlib import pyplot as plt
#################################################################################################

### PRESSURE FUNCTIONS ###

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
    dPdt = - a * q - b * (P - p0) - b * (P - p1)

    # Return derivative
    return dPdt

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

# Pressure ODE Solver and Grapher Function
def plot_aquifer_pressure(t0, t1, dt, t_q_data, q_data, t_p_data, p_data, P_parameters):
    ''' 
        Plot the kettle LPM over top of the data.

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

# Pressure Solution Evaluator Function (to pass into curve_fit during model calibration)
def evaluate_pressure(f, t0, t1, dt, t_q_data, q_data):

    def fit_pressure(t, *P_parameters):
        t_sol, P_sol = solve_ode_pressure(f, t0, t1, dt, t_q_data, q_data, P_parameters)
        return np.interp(t, t_sol, P_sol)

    return fit_pressure


