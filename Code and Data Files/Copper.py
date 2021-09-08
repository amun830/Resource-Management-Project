# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from Helper_Func import *
#################################################################################################

### COPPER CONCENTRATION FUNCTIONS ###

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
    dCdt = (-(b/a) * (P - p0) * (Cdash - C) + (b/a) * (P - p1) * C - dC_src * (P - 0.5 * (p0 + p1)))/M0

    # Return derivative
    return dCdt

# Copper Concentration ODE Solver Function
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

# Copper Concentration ODE Solver and Grapher Function
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

# Copper Concentration Solution Evaluator Function (to pass into curve_fit during model calibration)
def evaluate_copper(f, t0, t1, dt, t_sol, P_sol, theta_P):

    def fit_copper(t, *Extra_C_parameters):
        C_parameters = get_parameter_set(theta_P, Extra_C_parameters, "cu_all")
        t_cu_sol, cu_sol = solve_ode_cu(f, t0, t1, dt, t_sol, P_sol, C_parameters)
        return np.interp(t, t_cu_sol, cu_sol)

    return fit_copper


