from matplotlib.collections import PatchCollection
import pypolycontain as pp
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from matplotlib.patches import Polygon as poly
from utils.map import SinD_map


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
    _top = np.concatenate((z1.G, np.zeros(shape=z1.G.shape)), axis=1)
    _bottom = np.concatenate((np.zeros(shape=z2.G.shape), z2.G), axis=1)
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
    assert L.shape[1] == (z.x.shape[0] and z.G.shape[0])
    return pp.zonotope(x=L*z.x, G=L*z.G)


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
    _poly = Polygon(pp.to_V(z)).buffer(2*np.finfo(float).eps)
    return _poly.contains(Point(point))


def visualize_zonotopes(z: list, map: SinD_map = None, show: bool = False) -> None:
    """ Visualize zonotopes

        Parameters:
        -----------
        z : list(pp.zonotope)
            The z-parameter contain all the zonotopes that is 
            going to be visualized
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
    visualize(z, ax=map_ax, title="Zonotope visualization")
    if show: plt.show()





"""
    The following code is taken from pypolycontain in order to enable
    plotting the map and zonotopes without the axes automatically re-
    sizing according to the zonotopes. This way, the map determines the 
    absolute size of the figure and the zonotopes are simply plotted
    within the map
"""
def _projection(P,tuple_of_projection_dimensions):
    p_matrix=np.zeros((2,P.n))
    p_matrix[0,tuple_of_projection_dimensions[0]]=1
    p_matrix[1,tuple_of_projection_dimensions[1]]=1
    return pp.affine_map( p_matrix, P)


def visualize(list_of_objects,ax=None,alpha=0.8,tuple_of_projection_dimensions=[0,1],\
              title=r'pypolycontain visualization',\
              show_vertices=False,TitleSize=15,FontSize=15,equal_axis=False,grid=True,\
              N_points=1000):
    r"""
    Visualization.
    
    inputs: 
        * list_of_objects:
        * fig:
        * tuple_of_projection_dimensions: 
        * 
    """
    if type(ax)==type(None):
        _,ax=plt.subplots()
    #fig.set_size_inches(figsize[0],figsize[1])
    p_list,x_all=[],np.empty((0,2))  
    for p in list_of_objects:
        if p.n>2:
            print('projection on ',tuple_of_projection_dimensions[0],\
                  ' and ',tuple_of_projection_dimensions[1], 'dimensions')
            p=_projection(p,tuple_of_projection_dimensions)
        x=pp.to_V(p,N=N_points)
        mypolygon=poly(x)
        p_list.append(mypolygon) 
        x_all=np.vstack((x_all,x))
        if show_vertices:
            ax.plot(x[:,0],x[:,1],'*',color=p.color)
    p_patch = PatchCollection(p_list,color=[p.color for p in list_of_objects], alpha=alpha)
    ax.add_collection(p_patch)
    #ax.set_xlim([np.min(x_all[:,0])-a,a+np.max(x_all[:,0])])
    #ax.set_ylim([np.min(x_all[:,1])-a,a+np.max(x_all[:,1])])
    if grid:
        ax.grid(color=(0,0,0), linestyle='--', linewidth=0.3)
    ax.set_title(title,FontSize=TitleSize)
    ax.set_xlabel(r"$x_{%d}$"%(tuple_of_projection_dimensions[0]+1),FontSize=FontSize)
    ax.set_ylabel(r"$x_{%d}$"%(tuple_of_projection_dimensions[1]+1),FontSize=FontSize)
    if equal_axis:
        ax.axis('equal')

