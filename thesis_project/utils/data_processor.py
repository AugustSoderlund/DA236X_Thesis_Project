import numpy as np


def process_data(data: np.ndarray, input_len: int):
    """ Subtract the mean from (x,y) from data

        Parameters:
        -----------
        data_traj : np.ndarray
            The dataset to be processed
        input_len : int
            Length of each trajectory
    """
    _processed_data = []
    for _d in data:
        _x, _y = _d[0:input_len], _d[input_len:2*input_len]
        _x, _y = _x-np.mean(_x), _y-np.mean(_y)
        _processed_data.append(np.array([*_x, *_y, *_d[2*input_len:]]))
    return _processed_data