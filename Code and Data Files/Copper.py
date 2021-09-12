########################################### Copper.py ###########################################
    # 1. ode_cu(), returns the derivative dC/dt at time, t, for given parameters.
    # 2. solve_ode_cu(), solves the Cu concentration ODE numerically (Improved Euler method).
    # 3. plot_aquifer_cu(), solves the Cu conc. part of the LPM and plots over the data.
    # 4. evaluate_copper(), copper solution helper function for use in scipy.optimize.curve_fit.


# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from Helper_Func import *


#################################################################################################

# 1.

# Copper Concentration ODE Evaluator Function
def ode_cu(t, P, C, a, b, dC_src, p0, p1, M0):
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
        dC_src :float
                Copper concentration at surface source, as mass fraction (unitless) times by source/sink strength d (unitless).
        p0 :    float
                Ambient pressure at low pressure boundary (in Pa)
        p1 :    float
                Ambient pressure at high pressure boundary (in Pa)
        M0 :    float
                Mass of aquifer system (in kg).
       
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
    dCdt = (-(b/a) * (P - p0) * (Cdash - C) + (b/a) * (P - p1) * C - dC_src * (P - 0.5 * (p0 + p1)))/M0

    # Return derivative
    return dCdt


#################################################################################################

# 2.

# Copper Concentration ODE Solver Function
def solve_ode_cu(f, t0, t1, dt, t_sol, P_sol, C_parameters):
    ''' 
        Solve Cu conc. ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dCdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        t_sol : array_like
            List of time values from pressure solution.
        P_sol : array_like
            List of pressure values from pressure solution.
        C_parameters : array-like
            List of parameters passed to ODE function f, in order [a, b, dC_src, p0, p1, c_init, M0].

        Returns:
        --------
        t : array-like
            Time variable solution vector (years).
        C : array-like
            Cu concentration variable solution vector (mass fraction).

    '''
    
    # Unpack parameters
    [a, b, dC_src, p0, p1, c_init, M0] = C_parameters

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
        edxdt = f(t[i], p[i], C[i], a, b, dC_src, p0, p1, M0)
        ex1 = C[i] + dt*edxdt
        # Compute IE gradient
        iedxdt = f(t[i + 1], p[i+1], ex1, a, b, dC_src, p0, p1, M0)
        # Compute and store IE estimate of copper concentration
        C[i+1] = C[i] + 0.5 * dt * (edxdt + iedxdt)

    # Return time and copper concentration solution vectors
    return t, C


#################################################################################################

# 3.

# Copper Concentration ODE Solver and Grapher Function
def plot_aquifer_cu(t0, t1, dt, t_sol, P_sol, t_cu_data, cu_data, C_parameters):
    ''' 
        Plot the Cu conc. solution over the input time domain.

        Parameters:
        -----------
        t0 : float
            Initial time of solution (year)
        t1 : float
            Final time of solution (year)
        dt : float
            Time step length (in years)
        t_sol : array_like
            List of time values from pressure solution.
        P_sol : array_like
            List of pressure values from pressure solution.
        t_cu_data : array-like
            List of time points of given Cu concentration historical data (years) 
        cu_data : array-like
            List of Cu concentration historical data.
        C_parameters : array-like
            List of parameters passed to ODE function f, in order [a, b, dC_src, p0, p1, c_init, M0].

        Returns:
        --------
        m_time : array-like
            List of time points of pressure LPM data.
        m_press : array-like
            List of Cu concentration values of pressure LPM data.

        Notes:
        --------
        Plot also generated
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

# 4.

# Copper Concentration Solution Evaluator Function (to pass into curve_fit during model calibration)
def evaluate_copper(f, t0, t1, dt, t_sol, P_sol, theta_P):
    ''' 
        Cu concentration solution helper function for use in scipy.optimize.curve_fit.

        Parameters:
        -----------
        f : callable
            Function that returns dCdt given variable and parameter inputs.
        t0 : float
            Initial time of solution (year)
        t1 : float
            Final time of solution (year)
        dt : float
            Time step length (in years)
        t_sol : array-like
            List of time points of solved pressure part of LPM 
        P_sol : array-like
            List of pressure values of solved part of LPM
        theta_P : array-like
            List of parameters passed to ODE function f: a, b, p0, p1, p_init (in order)

        Returns:
        --------
        fit_copper : callable
            Function to allow scipy.optimize.curve_fit() to callibrate optimum parameter values for Cu concentration LPM
    '''

    def fit_copper(t, *Extra_C_parameters):
        C_parameters = get_parameter_set(theta_P, Extra_C_parameters, "cu_all")             # Uses Helper_func.py to arange parameters
        t_cu_sol, cu_sol = solve_ode_cu(f, t0, t1, dt, t_sol, P_sol, C_parameters)          # Computes copper solution
        return np.interp(t, t_cu_sol, cu_sol)                                               # Interpolates copper value at requried time input

    return fit_copper


