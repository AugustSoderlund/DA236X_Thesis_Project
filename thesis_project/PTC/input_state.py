import pypolycontain as pp
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from typing import List, Union
from utils.data_reader import LABELS
import random

if __package__ or "." in __name__:
    from utils.map import SinD_map
    from utils.poly_process import crosswalk_poly_for_label as cpfl
else:
    from ..utils.map import SinD_map
    from ..utils.poly_process import crosswalk_poly_for_label as cpfl



def separate_data_to_class(data: np.ndarray, classification: np.ndarray):
    """ Separate the entire dataset into a list[list] where each nested list contain 
        the trajectories for that specific class.

        Parameters:
        -----------
        data : np.ndarray
            The entire dataset
        classification : np.ndarray
            The labels from the classification framework for the dataset in data
    """
    _class = [0] * len(LABELS)
    for _i in range(len(_class)): _class[_i] = []
    for i,_trajectory in enumerate(data):
        _class[classification[i]].append(_trajectory)
    return _class

def create_io_state(data: List[np.ndarray], measurement: pp.zonotope, vel: np.ndarray, classification: Union[int, List[int]], input_len: int = 30) -> List[np.ndarray]:
    # TODO : Change such that this returns a dictionary with the input-state trajectories for each class separately that are near the pedestrian
    # TODO : Maybe remove the angle constraint and implement 'forge_traj' in here instead of at DRA.operations
    # TODO : Maybe remove points that are "behind" the pedestrian's zonotope
    """ Function to create D = (X-, X+, U-) in the reachability algorithm

        Parameters:
        -----------
        data : np.ndarray
            Data from that has been precomputed by the separate_data_to_class function
        measurement : pp.zonotope
            The measurement from which the reachable sets should be calculated
        vel : np.ndarray
            Current velocity vector of measurement
        classification : int | List[int]
            The classification for the current pedestrian as an int corresponding 
            to the class OR the list of all possible classes, in which case the
            function returns all trajectories near the pedestrian regardless of
            class
        input_len : int (default = 30)
            The input length for the chunks of each trajectory
    """
    if type(classification) == list: _data = np.concatenate(data)
    else: _data = data[classification]
    X_m, X_p, U = np.array([]), np.array([]), np.array([])
    _ped_poly = Polygon(pp.to_V(measurement))
    _angle_set = np.array([np.arctan2(*vel)-np.pi/8, np.arctan2(*vel)+np.pi/8])
    for _t in _data:
        _x, _y = _t[0:input_len], _t[input_len:2*input_len]
        _vx, _vy = _t[2*input_len:3*input_len], _t[3*input_len:4*input_len]
        _v_avg = np.array([sum(_vx[0:3]), sum(_vy[0:3])])
        _line = LineString(list(zip(_x, _y)))
        if _line.intersects(_ped_poly) and __in_between(np.arctan2(*_v_avg), _angle_set):
            _X = np.array([_x, _y])
            _X_p, _X_m = _X[0:,1:], _X[0:,:-1]
            _vx, _vy = _t[2*input_len:3*input_len], _t[3*input_len:4*input_len]
            _U = np.array([_vx, _vy])
            _U = _U[0:,:-1]
            X_p = np.hstack([X_p, _X_p]) if X_p.size else _X_p
            X_m = np.hstack([X_m, _X_m]) if X_m.size else _X_m
            U = np.hstack([U, _U]) if U.size else _U
    #U, X_p, X_m = U.reshape(-1,2,input_len-1), X_p.reshape(-1,2,input_len-1), X_m.reshape(-1,2,input_len-1)
    #U, X_p, X_m = sum(U)/U.shape[0], sum(X_p)/X_p.shape[0], sum(X_m)/X_m.shape[0]
    U = __drop_equal(U)
    X_p = __drop_equal(X_p)
    X_m = __drop_equal(X_m)
    return [U, X_p, X_m]

def __in_between(val: float, range: np.ndarray):
    assert range.shape[0] == 2
    return (val > range[0] and val < range[1])

def __drop_equal(arr: np.ndarray):
    _d, _ids = {}, np.array([], dtype=int)
    assert arr.shape[1] > arr.shape[0]
    for i, a in enumerate(arr.T):
        if str(a) not in _d:
            _d.update({str(a):0})
        elif str(a) in _d:
            _ids = np.hstack((_ids, i))
    return np.delete(arr, _ids, axis=1)

# def generate_trajectories(map: SinD_map, num_of_trajs: int, measurement: pp.zonotope, vel: np.ndarray, input_len: int = 30, dt: float = 0.1):
#     def add_noise(noise: float = 0.005):
#         return random.uniform(-noise, noise)
#     _p = Point(measurement.x)
#     _areas = [map.crosswalk_poly, map.sidewalk_poly, map.gap_poly, map.intersection_poly, map.road_poly]
#     _crosswalks = cpfl(map)
#     _trajs = {}.fromkeys(LABELS.keys())
#     [_trajs.update({l:[]}) for l in _trajs.keys()]
#     if _p.within(map.intersection_poly) or _p.within(map.road_poly) or _p.within(map.gap_poly):
#         noise = 0.009
#         x, y = [_p.x], [_p.y]
#         vx, vy = [vel[0]], [vel[1]]
#         for i in range(1,input_len*10):
#             x.append(x[i-1] + dt * vx[i-1] + add_noise(noise))
#             y.append(y[i-1] + dt * vy[i-1] + add_noise(noise))
#             vx.append(vx[i-1] + add_noise(noise))
#             vy.append(vy[i-1] + add_noise(noise))
#         _trajs[LABELS["cross_illegal"]].append(np.hstack((x,y))) 
#     elif _p.within(map.sidewalk_poly):
#         _dist1, _dist2 = 10000, 10000
#         _id1, _id2 = None, None
#         for i, _c in enumerate(_crosswalks):
#             _d = _c.distance(_p)
#             if _d < _dist1:
#                 _dist1, _dist2 = _d, _dist1
#                 _id1, _id2 = i, _id1
#             elif _d > _dist1 and _d < _dist2:
#                 _dist2 = _d
#                 _id2 = i
#     elif _p.within(map.crosswalk_poly):
#         pass
#     else:
#         raise NameError


            


    

