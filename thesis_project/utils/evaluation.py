import numpy as np
import pypolycontain as pp
from shapely.geometry import Polygon
from DRA.operations import optimize_vertices
from typing import Union, List
import os


ROOT = os.getcwd() + "/thesis_project/.datasets"
RA_PATH = "/SinD/reachable_sets.pkl"
RAB_PATH = "/SinD/reachable_base_sets.pkl"

def zonotope_area(z: Union[pp.zonotope, Polygon], simplify: bool = False) -> float:
    """ Calculate the area of a 2D zonotope

        Parameters:
        -----------
        z : pp.zonotope | Polygon
            The zonotope/polygon representing the 
            final reachable set
    """
    if type(z) == pp.zonotope: z = Polygon(optimize_vertices(z, simplify=simplify))
    assert type(z) == Polygon
    return z.area

def PTC_acc_per_class() -> List[float]:
    pass



if __name__ == "__main__":
    x = np.array([0,0])
    G = np.array([[1,0],[0,2]])
    z = pp.zonotope(G,x)
    a = zonotope_area(z)
    print(a)