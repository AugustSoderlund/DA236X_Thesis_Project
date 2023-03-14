from typing import List, Union
from matplotlib.collections import PatchCollection
import pypolycontain as pp
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from matplotlib.patches import Polygon as poly
from scipy.spatial import ConvexHull
import alphashape
#from utils.map import SinD_map
import platform

if platform.system() == "Windows":
    use_pydrake = False
else:
    use_pydrake = True


def minkowski_sum(z1: pp.zonotope, z2: pp.zonotope) -> pp.zonotope:
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

def product(z1: pp.zonotope, z2: pp.zonotope) -> pp.zonotope:
    """ Calculates the cartesian product of two zonotopes

        Parameters:
        -----------
        z1 : pp.zonotope
        z2 : pp.zonotope
    """
    c_z = np.concatenate((z1.x, z2.x), axis=0)
    _top = np.concatenate((z1.G, np.zeros(shape=z2.G.shape)), axis=1)
    _bottom = np.concatenate((np.zeros(shape=z1.G.shape), z2.G), axis=1)
    G_z = np.concatenate((_top, _bottom), axis=0)
    assert c_z.shape[0] == G_z.shape[0]
    return pp.zonotope(x=c_z, G=G_z)

def linear_map(L: np.ndarray, z: pp.zonotope) -> pp.zonotope:
    """ Perform linear map of a zonotope

        Parameters:
        -----------
        L : int
        z : pp.zonotope
    """
    if type(L) == np.ndarray: assert L.shape[1] == (z.x.shape[0] and z.G.shape[0]); _z = pp.zonotope(x=np.matmul(L,z.x), G=np.matmul(L,z.G))
    elif type(L) == int: _z = pp.zonotope(x=L*z.x, G=L*z.G)
    return _z


def is_inside(z: pp.zonotope, point: np.ndarray) -> bool:
    """ Check if a point is inside a zonotope, z

        Parameters:
        -----------
        z : pp.zonotope
            The zonotope that will be checked against
        point : np.ndararay
            The point that will be checked against the 
            zonotope, z
    """
    if not use_pydrake: V = optimize_vertices(z) 
    else: V = pp.to_V(z)
    _poly = Polygon(V).buffer(2*np.finfo(float).eps)
    return _poly.contains(Point(point))

def optimize_vertices(z: pp.zonotope, n: int = 2, simplify: bool = True):
    """ Optimize the vertices calculations by creating an alphashape 
        from the n first generators and then "resetting" the centers 
        by the coordinates from the boudary of the simplified convex hull

        Parameters:
        -----------
        z : pp.zonotope
            The zonotope from which the vertices should be calculated
        n : int (default = 2)
            After n generators create the alphashape and get the new
            points that should be used for continuing calculating the
            vertices
        simplify : bool (default = True)
            Simplify the alphashape. Setting this to True will give a
            more accurate description of the vertices but at the cost
            of increased computational time
    """
    if use_pydrake: return pp.to_V(z)
    else:
        c_z = np.copy(z.x)
        for i, g_z in enumerate(z.G.T):
            c_z = np.vstack([c_z+g_z, c_z-g_z])
            if i > 0 and i % n == 0:
                _shape = alphashape.alphashape(c_z, alpha=0.02).convex_hull
                if simplify: _shape = _shape.simplify(tolerance=0.05)
                _x, _y = _shape.boundary.xy
                _x, _y = np.array(_x), np.array(_y)
                c_z = np.vstack([_x,_y]).T
        try:
            return c_z[ConvexHull(c_z).vertices,:]
        except Exception as e:
            print(e)


def compute_vertices(z: pp.zonotope):
    """ (DEPRECATED)

        Compute the vertices of a zonotope

        Parameters:
        -----------
        z : pp.zonotope
            The zonotope from which the vertices should
            be calculated
    """
    c_z = np.copy(z.x)
    for g_z in z.G.T:
        c_z = np.vstack([c_z+g_z, c_z-g_z])
        print(c_z.shape)
    try:
        return c_z[ConvexHull(c_z).vertices], c_z
    except Exception as e:
        print(e)
        


def visualize_zonotopes(z: Union[List[pp.zonotope], List[np.ndarray]], map = None, show: bool = False, scale_axes: bool = False) -> None:
    """ Visualize zonotopes

        Parameters:
        -----------
        z : List[pp.zonotope] | List[np.ndarray]
            The z-parameter contain all the zonotopes that is 
            going to be visualized OR a list of vertices for
            all zonotopes that is to be plotted
        map : SinD_map (default = None)
            The map used for overlaying the zonotopes, if set
            to None the map will not be showed and only the
            zonotopes will be visible
        show : bool (default = False)
            Determines if the plot should be shown directly 
            after plotting of if user writes plt.show() in
            script where function is used
    """
    map_ax = None
    if map: map_ax = map.plot_areas()
    visualize(z, ax=map_ax, title="Zonotope visualization", scale_axes=scale_axes)
    if show: plt.show()





"""
    The following code is taken from pypolycontain in order to enable
    plotting the map and zonotopes without the axes automatically re-
    sizing according to the zonotopes. This way, the map determines the 
    absolute size of the figure and the zonotopes are simply plotted
    within the map
"""
def visualize(list_of_objects: Union[List[pp.zonotope], List[np.ndarray]], ax: plt.axes = None, alpha: float = 0.8, title: str = r'pypolycontain visualization', 
              show_vertices: bool = False, TitleSize: int = 15, FontSize: int = 15, equal_axis: bool = False, grid: bool = True,
              N_points: int = 1000, scale_axes: bool = False):
    r"""
    Visualization.
    
    inputs: 
        * list_of_objects:
        * fig:
        * tuple_of_projection_dimensions: 
        * 
    """
    a = 0.5
    if type(ax)==type(None):
        _,ax=plt.subplots()
    p_list, x_all= [], np.empty((0,2))  
    for p in list_of_objects:
        if type(p) == pp.zonotope:
            if use_pydrake: x = pp.to_V(p,N=N_points)
            else: x = optimize_vertices(p)
        else:
            x = p
        mypolygon = poly(x)
        p_list.append(mypolygon) 
        x_all = np.vstack((x_all, x))
        if show_vertices:
            ax.plot(x[:,0],x[:,1],'*',color=p.color)
    p_patch = PatchCollection(p_list, alpha=alpha)
    ax.add_collection(p_patch)
    if scale_axes:
        ax.set_xlim([np.min(x_all[:,0])-a,a+np.max(x_all[:,0])])
        ax.set_ylim([np.min(x_all[:,1])-a,a+np.max(x_all[:,1])])
    if grid:
        ax.grid(color=(0,0,0), linestyle='--', linewidth=0.3)
    ax.set_title(title, FontSize=TitleSize)
    ax.set_xlabel("x_1", FontSize=FontSize)
    ax.set_ylabel("x_2", FontSize=FontSize)
    if equal_axis:
        ax.axis('equal')

