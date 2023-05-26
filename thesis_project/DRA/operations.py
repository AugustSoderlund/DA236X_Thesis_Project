from typing import List, Union
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import pypolycontain as pp
from DRA.zonotope import zonotope, matrix_zonotope
import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from matplotlib.patches import Polygon as poly
from scipy.spatial import ConvexHull
import alphashape
import platform
from tqdm import tqdm

if __package__ or "." in __name__:
    from utils.map import SinD_map
else:
    from ..utils.map import SinD_map

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

def cartesian_product(z1: pp.zonotope, z2: pp.zonotope) -> pp.zonotope:
    """ Calculates the cartesian product of two zonotopes

        Parameters:
        -----------
        z1 : pp.zonotope
        z2 : pp.zonotope
    """
    z1.G = z1.G.reshape(z1.G.shape[0], -1)
    z2.G = z2.G.reshape(z2.G.shape[0], -1)
    c_z = np.concatenate((z1.x, z2.x), axis=0).reshape(z1.x.shape[0]+z2.x.shape[0], 1)
    _top = np.concatenate((z1.G, np.zeros(shape=(z1.G.shape[0], z2.G.shape[1]))), axis=1)
    _bottom = np.concatenate((np.zeros(shape=(z2.G.shape[0], z1.G.shape[1])), z2.G), axis=1)
    G_z = np.concatenate((_top, _bottom), axis=0)
    assert c_z.shape[0] == G_z.shape[0]
    return pp.zonotope(x=c_z, G=G_z)

def product(f1: Union[pp.zonotope, np.ndarray], f2: Union[pp.zonotope, np.ndarray]) -> pp.zonotope:
    """ Multiplies the zonotope with a factor (note: this is different
        from the cartesian product AND type of f1 and f2 can't both 
        be numpy.ndarray)

        Parameters:
        -----------
        f1 : pp.zonotope | np.ndarray
        f2 : pp.zonotope | np.ndarray
    """
    G_z = []
    if type(f1) == np.ndarray and type(f2) == np.ndarray: raise TypeError
    if type(f1) == type(f2):
        c_z = np.matmul(f1.x, f2.x)
        n = c_z.shape[0]
        G_z1 = np.matmul(f1.x, f2.G)                                 #  --> (2x5)
        G_z2 = np.matmul(f1.G, f2.x).reshape(n,-1)  
        G_z3 = np.matmul(f1.G, f2.G).reshape(n,-1)
        G_z = np.hstack((G_z1, G_z2, G_z3))
    elif type(f2) == pp.zonotope: 
        c_z = np.matmul(f1, f2.x) 
        n = c_z.shape[0]
        if len(f2.G.shape) == 1: G_z = np.matmul(f1, f2.G).reshape(f1.shape[0],-1)
        else: G_z = np.matmul(f1, f2.G).reshape(n,-1)#G_z = [[*G_z, *np.matmul(f1, g_z)] for g_z in f2.G]
    else:
        c_z = np.matmul(f1.x, f2) 
        G_z = [[*G_z, *np.matmul(g_z, f2)] for g_z in f1.G]
    try: 
        if c_z.shape[1] > 1: return matrix_zonotope(c_z, np.array(G_z))
        else: return zonotope(c_z, np.array(G_z))
    except Exception: return zonotope(c_z, np.array(G_z))



    
def create_M_w(total_size: int, Z: pp.zonotope = zonotope(np.array([0,0]), np.array([0.05,0.05])), disable_progress_bar: bool = False) -> pp.zonotope:
    """ Function to construct the matrix zonotope M_w

        Parameters:
        -----------
        total_size : int
            Total size of dataset X_
        Z : pp.zonotope
            Process noise zonotope
            dim(Z.G) -> (dim(c_z), 1)
    """
    _id = 0
    _G_w = [0] * (Z.G.shape[1] * total_size)
    for i in range(Z.G.shape[1]):
        _vec = Z.G[:,i].reshape(Z.G.shape[0], 1)
        _G_w[_id] = np.concatenate((_vec, np.zeros(shape=(Z.G.shape[0], total_size-1))), axis=1)
        for j in tqdm(range(1, total_size), desc="Creating noise zonotope", disable=disable_progress_bar):
            _G_w[_id+j] = np.concatenate((_G_w[_id+j-1][:,1:], _G_w[_id+j-1][:,0].reshape(Z.G.shape[0],1)), axis=1)
        _id = _id + j + 1
    return matrix_zonotope(C_M=Z.x, G_M=np.array(_G_w))



def linear_map(L: Union[int, np.ndarray], z: pp.zonotope, from_side: str = "left") -> pp.zonotope:
    """ Perform linear map of a zonotope

        Parameters:
        -----------
        L : int
        z : pp.zonotope
        from_side : str (default = 'left')
            Defines from which side the matrix L should be
            multiplied (from set {'left', 'right'}) 
    """
    if type(L) == np.ndarray: 
        if from_side == "left": assert L.shape[1] == (z.x.shape[0] and z.G.shape[0]); _z = pp.zonotope(x=np.matmul(L,z.x), G=np.matmul(L,z.G))
        elif from_side == "right": assert L.shape[0] == (z.x.shape[1] and z.G.shape[1]); _z = pp.zonotope(x=np.matmul(z.x,L), G=np.matmul(z.G,L))
        else: print("Only 'left' and 'right' are viable args"); raise TypeError
    elif type(L) == int or type(L) == float: _z = pp.zonotope(x=L*z.x, G=L*z.G)
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

def input_zonotope(U: List[np.ndarray], N: int = 30, gamma: str = "std") -> List[pp.zonotope]:
    """ Calculate the input zonotope U_k for the reachability analysis

        Parameters:
        ------------
        U : List[np.ndarray]
            The input pre-processed by the split_io_to_trajs function
        N : int (default = 30)
            Time horizon of reachability analysis
    """
    g, max_len = len(U), max(len(u[0,:]) for u in U)
    vx = np.ma.empty((g, max_len))
    vy = np.ma.empty((g, max_len))
    vx.mask, vy.mask = True, True
    for i,u in enumerate(U):
        #print(u, u.shape, u.shape[1], u[0,:].shape)
        vx[i,0:u.shape[1]] = u[0,:]
        vy[i,0:u.shape[1]] = u[1,:]
    vx_mean, vy_mean = vx.mean(axis=0), vy.mean(axis=0)
    if gamma == "std": vx_std, vy_std = vx.std(axis=0), vy.std(axis=0)
    elif gamma == "max": vx_std, vy_std = vx.max(axis=0), vy.max(axis=0)
    else: raise ValueError
    U_k = []
    for i in range(0,N):
        z = zonotope(c_z=np.array([vx_mean[i], vy_mean[i]]), G_z=np.array([[vx_std[i],0],[0,vy_std[i]]]))
        U_k.append(z)
    return U_k

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
        z = reduce(z, 6)
        return pp.to_V(z)
        """ LEGACY """
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

def reduce(z: pp.zonotope, order: int):
    """ Function that reduces the order of a zonotope 
        (NOTE: rewritten from MATLAB library 'CORA',
            Author: Matthias Althoff)

        Parameters:
        -----------
        z : pp.zonotope
        order : int
            How many generators should the reduced
            zonotope contain
    """
    def __non_zero_filter(G):
        _idx = np.argwhere(np.all(G[...,:] == 0, axis=0))
        return np.delete(G, _idx, axis=1)
    def __picked_generators(z: pp.zonotope, order: int):
        c_z, G = z.x, z.G
        G = __non_zero_filter(G)
        d, num_gens = G.shape
        if num_gens > d*order:
            h = np.linalg.norm(G,1,axis=0) - np.linalg.norm(G,np.inf,axis=0)
            n_unred = math.floor(d*(order-1))
            n_red = num_gens - n_unred
            _idx = np.argsort(h)[:n_red]
            G_red = G[:,_idx]
            _idy = np.setdiff1d(list(range(0,num_gens)), _idx)
            G_unred = G[:,_idy]
        else:
            G_red = np.array([])
            G_unred = G
        return (c_z, G_unred, G_red)
    def __girard(z: pp.zonotope, order: int):
        c_z, G_unred, G_red = __picked_generators(z, order)
        if G_red.size > 0: _G = np.diag(np.sum(abs(G_red), axis=1)); G = np.hstack((G_unred, _G))
        else: G = G_unred
        return zonotope(c_z=c_z, G_z=G)
    return __girard(z, order)



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
    try:
        return c_z[ConvexHull(c_z).vertices], c_z
    except Exception as e:
        print(e)
        


def visualize_zonotopes(z: Union[List[pp.zonotope], List[np.ndarray]], map: Union[SinD_map, plt.Axes] = None, show: bool = False, 
                        scale_axes: bool = False, plot_vertices: bool = True, _labels: list = None, _markers: list = None, title: str = "Zonotope visualization") -> plt.Axes:
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
    if type(map) == SinD_map: map_ax, _ = map.plot_areas()
    else: map_ax = map
    visualize(z, ax=map_ax, title=title, scale_axes=scale_axes, show_vertices=plot_vertices, _labels=_labels, markers=_markers)
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



__markers = ["o", "s", "x", "p", "v", "_", "|", "*"]
#_markers = ["*"] * 100
def visualize(list_of_objects: Union[List[pp.zonotope], List[np.ndarray]], ax: plt.Axes = None, alpha: float = 0.5, 
              title: str = r'pypolycontain visualization', show_vertices: bool = True, equal_axis: bool = False,
              grid: bool = False, scale_axes: bool = False, _labels: list = None, markers = None) -> plt.Axes:
    r"""
    Visualization.
    
    inputs: 
        * list_of_objects:
        * fig:
        * tuple_of_projection_dimensions: 
        * 
    """
    a = 0.5
    if type(markers) is np.ndarray: _markers = markers
    else: _markers = __markers
    tuple_of_projection_dimensions=[0,1]
    if type(ax)==type(None):
        _,ax=plt.subplots()
    p_list, x_all= [], np.empty((0,2))  
    for i,p in enumerate(list_of_objects):
        if p.n>2:
            print('projection on ',tuple_of_projection_dimensions[0],\
                  ' and ',tuple_of_projection_dimensions[1], 'dimensions')
            p=_projection(p,tuple_of_projection_dimensions)
        if type(p) == pp.zonotope:
            x = pp.to_V(reduce(p,6))
            #if use_pydrake: x = pp.to_V(p,N=N_points)
            #else: x = optimize_vertices(p)
        else:
            x = p
        mypolygon = poly(x)
        p_list.append(mypolygon) 
        x_all = np.vstack((x_all, x))
        if show_vertices:
            ax.plot(x[:,0],x[:,1],marker=_markers[i],color=p.color, markersize=4)
    p_patch = PatchCollection(p_list,color=[p.color for p in list_of_objects], alpha=alpha)
    ax.add_collection(p_patch)
    if scale_axes:
        ax.set_xlim([np.min(x_all[:,0])-a,a+np.max(x_all[:,0])])
        ax.set_ylim([np.min(x_all[:,1])-a,a+np.max(x_all[:,1])])
    if grid:
        ax.grid(color=(0,0,0), linestyle='--', linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if equal_axis:
        ax.axis('equal')
    if _labels:
        handles, labels = ax.get_legend_handles_labels()
        circs = [*handles]
        _labels_ = [*labels, *_labels]
        for i in range(len(list_of_objects)):
            circs.append(Line2D([0], [0], linestyle="none", marker=_markers[i], alpha=0.4, markersize=10, 
                                markerfacecolor=list_of_objects[i].color, markeredgecolor=list_of_objects[i].color))
        if len(list_of_objects) != 0: ax.legend(circs, _labels_, numpoints=1, loc="upper right")


