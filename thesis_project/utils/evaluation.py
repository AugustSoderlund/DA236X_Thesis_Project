import numpy as np
import pypolycontain as pp
from shapely.geometry import Polygon
from typing import Union, List


def zonotope_area(z: Union[pp.zonotope, Polygon]) -> float:
    """ Calculate the area of a 2D zonotope

        Parameters:
        -----------
        z : pp.zonotope | Polygon
            The zonotope/polygon representing the 
            final reachable set
    """
    if type(z) == pp.zonotope: z = Polygon(pp.to_V(z))
    assert type(z) == Polygon
    return z.area

def simulate_RA():
    pass

def PTC_acc_per_class() -> List[float]:
    pass

def AUC():
    pass


if __name__ == "__main__":
    x = np.array([0,0])
    G = np.array([[1,0],[0,2]])
    z = pp.zonotope(G,x)
    a = zonotope_area(z)
    print(a)