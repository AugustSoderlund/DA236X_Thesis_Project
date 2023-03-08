import pypolycontain as pp
import numpy as np
import matplotlib.pyplot as plt


def minkowski_sum(z1: pp.zonotope, z2: pp.zonotope):
    """ Perform the minkowski sum of two zonotopes

        Parameters:
        -----------
        z1 : pp.zonotope
        z2 : pp.zonotope
    """
    c_z = z1.x + z2.x
    G_z = np.concatenate((z1.G, z2.G), axis=1)
    assert c_z.shape[0] == G_z.shape[0]
    return pp.zonotope(x=c_z, G=G_z)

def product(z1: pp.zonotope, z2: pp.zonotope):
    """ Calculates the cartesian product of two zonotopes

        Parameters:
        -----------
        z1 : pp.zonotope
        z2 : pp.zonotope
    """
    c_z = np.concatenate((z1.x, z2.x), axis=0)
    _top = np.concatenate((z1.G, np.zeros(shape=z1.G.shape)), axis=1)
    _bottom = np.concatenate((np.zeros(shape=z2.G.shape), z2.G), axis=1)
    G_z = np.concatenate((_top, _bottom), axis=0)
    assert c_z.shape[0] == G_z.shape[0]
    return pp.zonotope(x=c_z, G=G_z)

def linear_map(L: int, z: pp.zonotope):
    """ Perform linear map of a zonotope

        Parameters:
        -----------
        L : int
        z : pp.zonotope
    """
    return pp.zonotope(x=L*z.x, G=L*z.G)

def visualize_zonotopes(z: list):
    """ Visualize zonotopes

        Parameters:
        -----------
        z : list(pp.zonotope)
            The z-parameter contain all the zonotopes that is 
            going to be visualized
    """
    pp.visualize(z, title="Zonotope visualization")
    plt.show()

