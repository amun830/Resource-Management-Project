######################################### Uncertainty_Func.py #########################################
    #1. create_posterior(), generates posterior distribution of LPM by calibrating off hystoric data.
    #2. plot_posterior2D(), plots posterior solution solved in 1.
    #3. fit_mvn(), calculates optimum parameters of a multivariate normal distribution.
    #4. construct_samples(), Constructs samples from a multivariate normal distribution fitted to the data.
    #5. plot_samples(), Plots ...
    #6. model_ensemble(), Runs the model for given parameter samples and plots using 5.


# Import libraries
import numpy as np
from Pressure import *
from Copper import *
from Helper_Func import *
from Model_Use_Func import *
import matplotlib
#################################################################################################

#1.

# Posterior Distribution Generator Function
def create_posterior(Parameters_best, N, quantity):
    '''
        Generates posterior distribution of LPM by calibrating off hystoric data.

        Parameters:
        -----------
        Parameters_best : array-like
            Parameters of LPM.
        N : Int
            ????
        quantity :  string
            Identifier, "pressure" or "copper".
       
        Returns:
        --------
        par1 :  array-like
            Vector of first parameter values.
        par2 :  array-like
            Vector of second parameter values.
        Posterior : ???
            ????
    
    '''

    # Unpack best estimates of parameter values
    a_best, b_best, p0_best, p1_best, p_init_best, dC_src_best, c_init_best, M0_best = get_parameter_set(Parameters_best, None, "get_all")

    # Generate vectors of parameter values
    if quantity == "pressure":
        par1 = np.linspace(a_best/2,a_best*1.6, N)              # a
        par2 = np.linspace(b_best/2,b_best*1.6, N)              # b
    elif quantity == "copper":
        par1 = np.linspace(dC_src_best/1.2,dC_src_best*1.2, N)  # dC_src
        par2 = np.linspace(M0_best/1.2,M0_best*1.2, N)          # M0

	# Create grid of all parameter combinations
    A, B = np.meshgrid(par1, par2, indexing='ij')
	# Initialise matrix for objective function
    S = np.zeros(A.shape)

    # Read in data for calibration
    if quantity == "pressure":
        t_data, data = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ').T
    elif quantity == "copper":
        t_data, data = np.genfromtxt("ac_cu.csv", dtype=float, skip_header=1, delimiter=', ').T


    # Set the error variance, at 0.005MPa for pressure and 0.02mg/L
    if quantity == "pressure":
        var = 4*10**-5
    elif quantity == "copper":
        var = 2*10**-5

    # Loop through all parameter combinations, and compute the sum of squares objective function for each
    for i in range(len(par1)):
        for j in range(len(par2)):
            if quantity == "pressure":
                model,_ = solve_lpm(t_data,[par1[i],par2[j],p0_best, p1_best, p_init_best, dC_src_best, c_init_best, M0_best])  
            elif quantity == "copper":
                _, model = solve_lpm(t_data,[a_best,b_best,p0_best, p1_best, p_init_best, par1[i], c_init_best, par2[j]])
            S[i,j] = np.sum((data-model)**2)/var


    # Compute the posterior
    Posterior = np.exp(-S/2.)

    # Normalise to a probability density function
    Volume = np.sum(Posterior) * (par1[1]-par1[0]) * (par2[1]-par2[0])
    Posterior /= Volume

    return par1, par2, Posterior


#################################################################################################

#2.

# Posterior Distribution Plotting Function
def plot_posterior2D(par1, par2, name1, name2, title, P):	
    """
        Plots posterior distribution


        Parameters:
        -----------
        par1 :  array-like
            Vector of first parameter values.
        par2 :  array-like
            Vector of second parameter values.
        name1 : string
            X label.
        name2 : string
            Y label.
        title : string
            Plot title.
        P : ???
            Posterior matrix.
       
        Returns:
        --------
        fig :   plt.figure()
            Plot of posterior distribution.
    
    """
    
    # grid of parameter values: returns every possible combination of parameters in a and b
    A, B = np.meshgrid(par1, par2)
    
    # plotting
    fig = plt.figure()				# open figure
    ax1 = fig.add_subplot(111, projection='3d')		# create 3D axes
    ax1.plot_surface(A, B, P, cmap=matplotlib.cm.Oranges,edgecolor='k')	# show surface
    
    # plotting upkeep
    ax1.set_xlabel(name1, fontsize = 11, color='c')
    ax1.set_ylabel(name2, fontsize = 11, color='c')
    ax1.set_zlabel('P', fontsize = 11, color='r')
    ax1.set_zlim(0., )
    ax1.set_title(title)
    
    return fig


#################################################################################################

#3.

def fit_mvn(parspace, dist):
    """Finds the parameters of a multivariate normal distribution that best fits the data

    Parameters:
    -----------
        parspace : array-like
            list of meshgrid arrays spanning parameter space
        dist : array-like 
            PDF over parameter space
    Returns:
    --------
        mean : array-like
            distribution mean
        cov : array-like
            covariance matrix		
    """
    
    # dimensionality of parameter space
    N = len(parspace)
    
    # flatten arrays
    parspace = [p.flatten() for p in parspace]
    dist = dist.flatten()
    
    # compute means
    mean = [np.sum(dist*par)/np.sum(dist) for par in parspace]
    
    # compute covariance matrix
        # empty matrix
    cov = np.zeros((N,N))
        # loop over rows
    for i in range(0,N):
            # loop over upper triangle
        for j in range(i,N):
                # compute covariance
            cov[i,j] = np.sum(dist*(parspace[i] - mean[i])*(parspace[j] - mean[j]))/np.sum(dist)
                # assign to lower triangle
            if i != j: cov[j,i] = cov[i,j]
            
    return np.array(mean), np.array(cov)


#################################################################################################

#4.

def construct_samples(par1,par2, Parameters, P,N_samples, quantity, plot):
	''' 
    This function constructs samples from a multivariate normal distribution
	fitted to the data.

	Parameters:
    -----------
		par1 :  array-like
            Vector of first parameter values.
        par2 :  array-like
            Vector of second parameter values.
        Parameters :    array-like
            Vector of LPM parameters.
		P : array-like
			Posterior probability distribution.
		N_samples : int
			Number of samples to take.
        quantity :  string
            Identifier ("copper" or "pressure")
        plot : boolean
            If true plot the samples

	Returns:
	--------
		samples : array-like
			parameter samples from the multivariate normal
	'''
	# compute properties (fitting) of multivariate normal distribution
	# mean = a vector of parameter means
	# covariance = a matrix of parameter variances and correlations
	A, B = np.meshgrid(par1,par2,indexing='ij')
	mean, covariance = fit_mvn([A,B], P)

	# 1. create samples using numpy function multivariate_normal (Google it)
	#samples=
	samples = np.random.multivariate_normal(mean, covariance, N_samples, tol=1e-5) 
	
	if plot == True:
		plot_samples(par1, par2, Parameters, P, samples, quantity) 
	
	return samples


#################################################################################################

#5.

def plot_samples(par1, par2, Parameters, P, samples, quantity):
    ''' 
    This function plots multivariate normal samples.

	Parameters:
    -----------
		par1 :  array-like
            Vector of first parameter values.
        par2 :  array-like
            Vector of second parameter values.
        Parameters :    array-like
            Vector of LPM parameters.
		P : array-like
			Posterior probability distribution.
		samples : array-like
			parameter samples from the multivariate normal
        quantity :  string
            Identifier ("copper" or "pressure")

	'''
    # plotting
    fig = plt.figure(figsize=[10., 7.])				# open figure
    ax1 = fig.add_subplot(111, projection='3d')		# create 3D axes
    A, B = np.meshgrid(par1, par2, indexing='ij')
    ax1.plot_surface(A, B, P, rstride=1, cstride=1,cmap=matplotlib.cm.Oranges, lw = 0.5)	# show surface
    
    tp, po = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ').T
    tc, co = np.genfromtxt("ac_cu.csv", dtype=float, skip_header=1, delimiter=', ').T

    if quantity == "pressure":
        v = 4*10**-5
        Other_Pars = Parameters[2:]
        s = np.array([np.sum((solve_lpm(tp,[par1,par2,*Other_Pars])[0]-po)**2)/v for par1,par2 in samples])
    elif quantity == "copper":
        v = 2*10**-5
        Before_Pars = Parameters[:5]
        Mid_Pars = Parameters[6]
        s = np.array([np.sum((solve_lpm(tc,[*Before_Pars, par1, Mid_Pars, par2])[1]-co)**2)/v for par1,par2 in samples])


    p = np.exp(-s/2.)
    p = p/np.max(p)*np.max(P)*1.2

    ax1.plot(*samples.T,p,'k.')

    # plotting upkeep
    if quantity == "pressure":
        ax1.set_xlabel('a (1/ms^2)', fontsize = 12, color='c')
        ax1.set_ylabel('b (1/s)', fontsize = 12, color='c')
    elif quantity == "copper":
        ax1.set_xlabel('dC_src (ms)', fontsize=12, color='c')
        ax1.set_ylabel('M0 (kg)', fontsize=12, color='c')
    ax1.set_zlim(0., )
    ax1.set_zlabel('P', fontsize=12, color='r')
    
    # save and show
    plt.show()


#################################################################################################

#6.

def model_ensemble(samples, Parameters,quantity):
    ''' 
    Runs the model for given parameter samples and plots the results.

	Parameters:
	-----------
	samples : array-like
	    parameter samples from the multivariate normal.
    Parameters :    array-like
        Vector of LPM parameters.
    quantity :  string
            Identifier ("copper" or "pressure")
    '''

    t = np.linspace(1980, 2018, 50)

    f,axis = plt.subplots(1,1)


    if quantity == "pressure":
        t_data, data = np.genfromtxt("ac_p.csv", dtype=float, skip_header=1, delimiter=', ').T
    elif quantity == "copper":
        t_data, data = np.genfromtxt("ac_cu.csv", dtype=float, skip_header=1, delimiter=', ').T


    for par1, par2 in samples:

        if quantity == "pressure":
            Parameters[0:2] = par1, par2
            pm = solve_lpm(t, Parameters)[0]
            axis.plot(t,pm,'k-',lw=0.25,alpha=0.4)
        elif quantity == "copper":
            Parameters[5] = par1
            Parameters[-1] = par2
            cum = solve_lpm(t, Parameters)[1]
            axis.plot(t,cum,'k-',lw=0.25,alpha=0.4)


    axis.plot([],[],'k-',lw=0.5,alpha=0.4,label="Model Ensemble")


    if quantity == "pressure":
        v = 4*10**-5
        axis.errorbar(t_data,data,yerr=v,fmt='ro', label=' Pressure data')
        axis.set_ylabel('Pressure (MPa)')
    elif quantity == "copper":
        v = 4*10**-5
        axis.errorbar(t_data,data,yerr=v,fmt='ro', label='Copper conc. data')
        axis.set_ylabel('Copper Concentration (mg/L)')
    
    axis.set_xlabel('Time (Year)')
    axis.legend()
    plt.show()





