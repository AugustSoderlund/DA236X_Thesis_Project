from dataclasses import dataclass
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


@dataclass
class Zonotope:
    c_z : np.ndarray
    G_z : np.ndarray

    def __post_init__(self):
        assert self.c_z.shape[0] == self.G_z.shape[0]

@dataclass
class MatrixZonotope:
    # TODO: Test this class to see if it works
    C_M : np.ndarray
    G_M : np.ndarray

    def __post_init__(self):
        assert self.C_M.shape[0] == self.G_M.shape[0]
        assert self.C_M.shape[1] == self.G_M[0].shape[0]



if __name__ == "__main__":
    x=np.array([1,1])
    G=np.array([[1,0,0],[0,1,0]])
    Zonotope(x, G)