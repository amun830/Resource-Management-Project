# Import libraries
import numpy as np
#################################################################################################

### HELPER FUNCTIONS ###

# Parameter Set Function (to return a required ordered set of parameters)
def get_parameter_set(list1, list2, type):

    if type == "theta_all":
        return np.concatenate([list1,list2])                                        # Return list of all calibrated model parameters

    elif type == "cu_all":
        [a, b, p0, p1, _] = list1           # theta_P
        [dC_src, c_init, M0] = list2      # Extra_C_parameters
        return [a, b, dC_src, p0, p1, c_init, M0]                                 # Return list of all copper model parameters
        
    elif type == "split" and list2 == None:
        [a, b, p0, p1, p_init, dC_src, c_init, M0] = list1
        return [a, b, p0, p1, p_init], [a, b, dC_src, p0, p1, c_init, M0]         # Return lists of each model parameters

    elif type == "get_all":
        [a, b, p0, p1, p_init, dC_src, c_init, M0] = list1
        return a, b, p0, p1, p_init, dC_src, c_init, M0

    else:
        return NameError                                                            # Else return name error


