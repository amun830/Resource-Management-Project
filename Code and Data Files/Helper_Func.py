########################################### Helper_Func.py ###########################################
    #1. Defines get_parameter_set(), which combines and arranges parameters into correct arangement.
        #1.1    type == "theta_all", returns list of all calibrated model parameters
        #1.2    type == "cu_all", returns list of all copper model parameters.
        #1.3    type == "split", splits list and return lists of each model parameters
        #1.4    type == "get_all", returns all parameters in list.




# Import libraries
import numpy as np

#################################################################################################

#1. 

# Parameter Set Function (to return a required ordered set of parameters)
def get_parameter_set(list1, list2, type):
    '''
     Return the derivative dC/dt at time, t, for given parameters.

        Parameters:
        -----------
        list1 : array-like
            First list of parameters.
        list2 : array-like
            Second list of parameters.
        type :  string
            Identifier of which method to use.
       
        Returns:
        --------
        _ : List
            Will return different objects depending on method used.
    '''

    #1.1
    if type == "theta_all":
        return np.concatenate([list1,list2])                                        # Return list of all calibrated model parameters

    #1.2
    elif type == "cu_all":
        [a, b, p0, p1, _] = list1           # theta_P
        [dC_src, c_init, M0] = list2      # Extra_C_parameters
        return [a, b, dC_src, p0, p1, c_init, M0]                                 # Return list of all copper model parameters
    
    #1.3
    elif type == "split" and list2 == None:
        [a, b, p0, p1, p_init, dC_src, c_init, M0] = list1
        return [a, b, p0, p1, p_init], [a, b, dC_src, p0, p1, c_init, M0]         # Return lists of each model parameters

    #1.4
    elif type == "get_all":
        [a, b, p0, p1, p_init, dC_src, c_init, M0] = list1
        return a, b, p0, p1, p_init, dC_src, c_init, M0

    else:
        return NameError                                                            # Else return name error


