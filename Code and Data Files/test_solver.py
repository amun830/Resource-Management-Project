# import libraries and functions
import numpy as np
from Pressure import *
from Copper import *
from numpy.linalg import norm

def test_solve_ode_pressure():
    """
    Test if function solve_ode_pressure is working properly by comparing it to a hand-solved result.
    
    """
    # Test 1
    P_parameters = [0.01, 0.2, 1.5, 1.6, 1.55]
    t_data = [0]
    q_data = [1]
    t_1, p_1 = solve_ode_pressure(ode_pressure, 0, 10, 1, t_data, q_data,P_parameters)
    p_1_10 = p_1[-1]
    p_1_10_soln = 1.525528480705039
    assert(norm(p_1_10 - p_1_10_soln) < 1.e-10)

    # Test 2
    P_parameters = [0.01, 0.02, 1.5, 1.6, 1.55]
    t_data = [0,10]
    q_data = t_data
    t_2, p_2 = solve_ode_pressure(ode_pressure, 0, 10, 1, t_data, q_data, P_parameters)
    p_2_10 = p_2[-1]
    p_2_10_soln = 1.110039186160613
    assert(norm(p_2_10 - p_2_10_soln) < 1.e-10)

    # Test 3
    P_parameters = [0, 0.02, 100, 100, 0]
    t_data = [0,10]
    q_data = t_data
    t_3, p_3 = solve_ode_pressure(ode_pressure, 0, 1, 1, t_data, q_data, P_parameters)
    p_3_10 = p_3[-1]
    p_3_10_soln = 3.92
    assert(norm(p_3_10 - p_3_10_soln) < 1.e-10)

    # Test 4
    P_parameters = [0, 0, 0, 0, 0]
    t_data = [0]
    q_data = [1]
    t_4, p_4 = solve_ode_pressure(ode_pressure, 0, 10, 1, t_data, q_data, P_parameters)
    p_4_10 = p_4[-1]
    p_4_10_soln = 0
    assert(norm(p_4_10 - p_4_10_soln) < 1.e-10)

test_solve_ode_pressure()