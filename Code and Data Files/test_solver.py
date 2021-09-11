# import libraries and functions
import numpy as np
from Pressure import *
from Copper import *
from numpy.linalg import norm

#
#
# TESTING PRESSURE ODE SOLVER
#
#

# Test 1
def test_solve_ode_pressure_1():
    """
    Test if function solve_ode_pressure is working properly by comparing it to a hand-solved result.
    Test should return valid.
    Testing concept of interest: Passing values for q such that the function uses a constant q value.
    """
    # Assign parameters
    P_parameters = [0.01, 0.2, 1.5, 1.6, 1.55]
    # Create data for q (to be interpolated) such that q is a constant value
    # q(t) = 1
    t_data = [0]
    q_data = [1]
    # Apply solver
    t_1, p_1 = solve_ode_pressure(ode_pressure, 0, 10, 1, t_data, q_data,P_parameters)
    # Extract last value for checking
    p_1_10 = p_1[-1]
    # Hand-solved result
    p_1_10_soln = 1.525528480705039
    # Perform comparison to see if results are the same
    assert(norm(p_1_10 - p_1_10_soln) < 1.e-8)

def test_solve_ode_pressure_2():
    """
    Test if function solve_ode_pressure is working properly by comparing it to a hand-solved result.
    Test should return valid.
    Testing concept of interest: Passing values for q such that the function uses a non-constant q value.
    """
    # Assign parameters
    P_parameters = [0.01, 0.02, 1.5, 1.6, 1.55]
    # Create data for q (to be interpolated) such that q is a non-constant value
    # q(t) = t
    t_data = [0,10]
    q_data = t_data
    # Apply solver
    t_2, p_2 = solve_ode_pressure(ode_pressure, 0, 10, 1, t_data, q_data, P_parameters)
    # Extract last value for checking
    p_2_10 = p_2[-1]
    # Hand-solved result
    p_2_10_soln = 1.110039186160613
    # Perform comparison to see if results are the same
    assert(norm(p_2_10 - p_2_10_soln) < 1.e-8)

def test_solve_ode_pressure_3():
    """
    Test if function solve_ode_pressure is working properly by comparing it to a hand-solved result.
    Test should return valid.
    Testing concept of interest: Passing all parameters as 0.
    """
    # Assign all parameters to be 0
    P_parameters = [0, 0, 0, 0, 0]
     # Create data for q (to be interpolated) such that q is a constant value of 0
    # q(t) = 0
    t_data = [0]
    q_data = [0]
    # Apply solver
    t_4, p_4 = solve_ode_pressure(ode_pressure, 0, 10, 1, t_data, q_data, P_parameters)
    # Extract last value for checking
    p_4_10 = p_4[-1]
    # Perform comparison to see if results are the same
    assert(norm(p_4_10 - 0) < 1.e-8)

def test_solve_ode_pressure_4():
    """
    Test if function solve_ode_pressure is working properly by comparing it to a hand-solved result.
    Test should return valid.
    Testing concept of interest: Giving data for q that only extends for part of the main data range.  
    """
    # Assign parameters
    P_parameters = [0.01, 0.2, 1.5, 1.6, 1.55]
    # Create data for q (to be interpolated) such that q is a non-constant value
    # Data starts at t=1 and ends at t=2, with values 0 and 1 respectively
    t_data = [1,2]
    q_data = [0,1]
    # Apply solver
    t_4, p_4 = solve_ode_pressure(ode_pressure, 0, 5, 0.5, t_data, q_data, P_parameters)
    # Extract value at t=1.5, (where interpolation will have taken place between q values)
    p_4_3 = p_4[3]
    # Hand-solved result
    p_4_3_soln = 1.54875
    # Perform comparison to see if results are the same
    assert(norm(p_4_3 - p_4_3_soln) < 1.e-8)

#
#
# TESTING COPPER CONCENTRATION ODE SOLVER
#
#

def test_solve_ode_cu_1():
    """
    Test if function solve_ode_cu is working properly by comparing it to a hand-solved result.
    Test should return valid.
    Testing concept of interest:  Passing values for pressure such that the function uses a constant value.
    """
    # Assign parameters
    C_parameters = [0.01, 0.2, 2, 1.5, 1.6, 2.5, 10]
    # Apply solver
    # Using constant value of 1 for pressure
    t_1, c_1 = solve_ode_cu(ode_cu, 0, 10, 1, [0], [1], C_parameters)
    # Extract last value for checking
    c_1_10 = c_1[-1]
    # Hand-solved result
    c_1_10_soln = 17.946346967798398
    # Perform comparison to see if results are the same
    assert(norm(c_1_10 - c_1_10_soln) < 1.e-8)

def test_solve_ode_cu_2():
    """
    Test if function solve_ode_cu is working properly by comparing it to a hand-solved result.
    Test should return valid.
    Testing concept of interest:  Passing values for pressure such that the function uses a non-constant value.
    """
    # Assign parameters
    C_parameters = [0.01, 0.2, 2, 1.5, 1.6, 2.5, 10]
    # Using non-constant value for pressure
    # p(t) = t
    t_2, c_2 = solve_ode_cu(ode_cu, 0, 10, 1, [0,10], [0,10], C_parameters)
    # Extract value at t=2
    c_2_2 = c_2[2]
    # Hand-solved result
    c_2_2_soln = -5.11902
    # Perform comparison to see if results are the same
    assert(norm(c_2_2 - c_2_2_soln) < 1.e-8)

def test_solve_ode_cu_3():
    """
    Test if function solve_ode_cu is working properly by comparing it to a hand-solved result.
    Test should return valid.
    Testing concept of interest: Passing all parameters as 0 (except for terms that are divisors).
    """
    # Assign parameters, a and M0 cannot be 0 therefore, are set to 1
    C_parameters = [1, 0, 0, 0, 0, 0, 1]
    # Apply solver
    # Using constant value of 1 for pressure
    t_3, c_3 = solve_ode_cu(ode_cu, 0, 10, 1, [0], [1], C_parameters)
    # Extract last value for checking
    c_3_10 = c_3[-1]
    # Perform comparison to see if results are the same
    assert(norm(c_3_10 - 0) < 1.e-8)

def test_solve_ode_cu_4():
    """
    Test if function solve_ode_cu is working properly by comparing it to a hand-solved result.
    Test should return valid.
    Testing concept of interest: Making pressure values with pressure solver first - constant q.
    """
    # Assign parameters for pressure
    P_parameters = [0.01, 0.2, 1.5, 1.6, 1.55]
    # Apply solver for pressure
    # Using constant value of 1 for q
    t_4, p_4 = solve_ode_pressure(ode_pressure, 0, 5, 0.5, [0], [1], P_parameters)
    # Assign parameters for copper concentration
    C_parameters = [0.01, 0.2, 2, 1.5, 1.6, 2.5, 10]
    # Apply solver for copper concentration
    # Using previously formed pressure values 
    t_4, c_4 = solve_ode_cu(ode_cu, 0, 10, 1, t_4, p_4, C_parameters)
    # Extract value at t=2 for checking
    c_4_2 = c_4[2]
    # Hand-solved result
    c_4_2_soln = 1.98999436
    # Perform comparison to see if results are the same
    assert(norm(c_4_2 - c_4_2_soln) < 1.e-8)

def test_solve_ode_cu_5():
    """
    Test if function solve_ode_cu is working properly by comparing it to a hand-solved result.
    Test should return valid.
    Testing concept of interest: Making pressure values with pressure solver first - non-constant q.
    """
    # Assign parameters for pressure
    P_parameters = [0.01, 0.2, 1.5, 1.6, 1.55]
    # Apply solver for pressure
    # Using values for q with smaller time-range
    t_5, p_5 = solve_ode_pressure(ode_pressure, 0, 5, 0.5, [1,2], [1,2], P_parameters)
    # Assign parameters for copper concentration
    C_parameters = [0.01, 0.2, 2, 1.5, 1.6, 2.5, 10]
    # Apply solver for copper concentration
    t_5, c_5 = solve_ode_cu(ode_cu, 0, 10, 1, t_5, p_5, C_parameters)
    # Extract value at t=2 for checking
    c_5_2 = c_5[2]
    # Hand-solved result
    c_5_2_soln = 1.981463523
    # Perform comparison to see if results are the same
    assert(norm(c_5_2 - c_5_2_soln) < 1.e-8)