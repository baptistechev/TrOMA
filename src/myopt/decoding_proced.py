import copy
from collections import Counter
import numpy as np

def Matchpurising_explicit(marginals,sketch,step,iteration_number, opt_func=np.argmax):
    """
    Perform matching pursuit to find a sparse solution to the linear system defined by the sketch matrix and the marginals.

    Parameters
    ----------
    marginals : list of float
        The marginals of the function defined on the full spectrum of dit strings. The order of the values should correspond
        to the order of the dit strings in the full spectrum.
    sketch : 2D numpy array
        The sketch matrix, where each column corresponds to a dit string and each row corresponds to a marginal.
    step : float
        The step size for the matching pursuit algorithm.
    iteration_number : int
        The number of iterations to perform in the matching pursuit algorithm.
    opt_func : function, optional
        A function that takes a 1D numpy array as input and returns the index of the maximum value. This function is used to select the column of the sketch matrix that best matches the current residual. The default is np.argmax.
    
    Returns
    -------
    numpy array
        A 2D numpy array where each row contains the index of a selected column from the sketch matrix and its corresponding coefficient in the sparse solution.
    """

    #Matching Pursuit, initialization
    r = copy.deepcopy(marginals)              
    x = []

    for _ in range(iteration_number):
        
        #Find max
        t = opt_func(sketch.T * r)

        #Update
        r -= step * sketch[:,t]
        x.append(t)
    
    return np.array([[elem[0],elem[1]*step] for elem in Counter(x).items()])


def Matchpurising_abstract(y,patterns,step,itnum,number_spins, spin_dimension=2,size_nn_pattern=2):
#Return x st y=Ax

    #Matching Pursuit, initialization
    r = copy.deepcopy(y)              
    x = []

    for _ in range(itnum):
        
        #Find max
        conf = spin_chain_pairs_max(marginals=r, number_spins=number_spins, spin_dimension=spin_dimension).astype(int).tolist()
        t = config_to_integer_position_2pow37([0]+conf)
        At = np.array(reconstruct_column(t,patterns))

        #Update
        r -= step * At 
        x.append(t)
    
    return np.array([[it[0],it[1]*step] for it in Counter(x).items()])