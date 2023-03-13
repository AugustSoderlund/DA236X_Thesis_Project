import pypolycontain as pp
import numpy as np
from shapely.geometry import Polygon, LineString
from typing import List
from utils.data_reader import LABELS



def separate_data_to_class(data: np.ndarray, classification: np.ndarray, input_len: int = 30):
    """ Separate the entire dataset into a list[list] where each nested list contain 
        the trajectories for that specific class.

        Parameters:
        -----------
        data : np.ndarray
            The entire dataset
        classification : np.ndarray
            The labels from the classification framework for the dataset in data
        input_len : int (default = 30)
            The input length for the chunks of each trajectory
    """
    _class = [[]] * len(LABELS)
    for i,_trajectory in enumerate(data):
        _t = _trajectory[0:2*input_len]
        _class[classification[i]].append(_t)
    return _class

def create_io_state(data: np.ndarray, measurement: pp.zonotope, classification: int, input_len: int = 30) -> List[np.ndarray]:
    """ Function to create D = (X-, X+, U-) in the reachability algorithm

        Parameters:
        -----------
        data : np.ndarray
            Data from that has been precomputed by the separate_data_to_class function
        measurement : pp.zonotope
            The measurement from which the reachable sets should be calculated
        classification : int
            The classification for the current pedestrian as an int corresponding 
            to the class
        input_len : int (default = 30)
            The input length for the chunks of each trajectory
    """
    X_m, X_p, U = np.array([]), np.array([]), np.array([])
    _ped_poly = Polygon(pp.to_V(measurement))
    _data = data[classification]
    for _t in _data:
        _x, _y = _t[0:input_len], _t[input_len:2*input_len]
        _line = LineString(list(zip(_x, _y)))
        if _line.within(_ped_poly):
            _X = np.array([_x, _y])
            _X_p, _X_m = _X[0:,1:], _X[0:,:-1]
            _vx, _vy = _t[2*input_len:3*input_len], _t[3*input_len:4*input_len]
            _U = np.array([_vx, _vy])
            _U = _U[0:,:-1]
            X_p = np.hstack([X_p, _X_p]) if X_p.size else X_p
            X_m = np.hstack([X_m, _X_m]) if X_m.size else X_m
            U = np.hstack([U, _U]) if U.size else U
    return [U, X_p, X_m]
            


    

