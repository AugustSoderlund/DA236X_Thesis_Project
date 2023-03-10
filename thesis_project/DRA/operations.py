import pypolycontain as pp
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon


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

def linear_map(L, z: pp.zonotope):
    """ Perform linear map of a zonotope

        Parameters:
        -----------
        L : int
        z : pp.zonotope
    """
    # TODO: Implement some asserts here depending on type(L)
    return pp.zonotope(x=L*z.x, G=L*z.G)


def is_inside(z: pp.zonotope, point: np.ndarray):
    """ Check if a point is inside a zonotope, z

        Parameters:
        -----------
        z : pp.zonotope
            The zonotope that will be checked against
        point : np.ndararay
            The point that will be checked against the 
            zonotope, z
    """
    _poly = Polygon(pp.to_V(z)).buffer(2*np.finfo(float).eps)
    return _poly.contains(Point(point))


def visualize_zonotopes(z: list, show: bool = False):
    """ Visualize zonotopes

        Parameters:
        -----------
        z : list(pp.zonotope)
            The z-parameter contain all the zonotopes that is 
            going to be visualized
    """
    pp.visualize(z, title="Zonotope visualization")
    if show: plt.show()

