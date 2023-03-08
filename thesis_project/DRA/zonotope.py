import pypolycontain as pp
import numpy as np

def zonotope(c_z: np.ndarray, G_z: np.ndarray):
    """ Zonotope creation 

        Parameters:
        -----------
        c_z : np.ndarray
            The center of the measurement
        G_z : np.ndarray    
            Generator matrix for zonotope
    """
    assert c_z.shape[0] == G_z.shape[0]
    return pp.zonotope(x=c_z, G=G_z)