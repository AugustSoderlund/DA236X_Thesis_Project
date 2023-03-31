from itertools import repeat
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pypolycontain as pp
import multiprocessing
from multiprocessing import Pool
import imageio



if __package__ or "." in __name__:
    from utils.map import SinD_map
    from utils.data_reader import SinD, LABELS
    from PTC.dnn_classifier import DNN
    from DRA.operations import visualize_zonotopes, input_zonotope, create_M_w
    from DRA.reachability import LTI_reachability
    from PTC.input_state import create_io_state, separate_data_to_class, split_io_to_trajs
    from DRA.zonotope import zonotope
    from utils.data_processor import load_data, structure_input_data, split_data
else:
    from map import SinD_map
    from ..DRA.operations import visualize_zonotopes
    from ..DRA.reachability import LTI_reachability

ROOT = os.getcwd() + "/thesis_project/.datasets"
DATADIR = "SinD/Data"
DATASET = "8_02_1"
RA_PATH = "/SinD/reachable_sets.pkl"

def load_data_for_simulation(dnn: DNN, name: str = "Ped_smoothed_tracks.csv", input_len: int = 90, load_data: bool = False):
    """ Load the dataset in such way that it can be simulated
        with appropriate frame appearances from pedestrians

        Parameters:
        -----------
        name : str (default = 'Ped_smoothed_tracks.csv')
    """
    _path = "/".join([ROOT, DATADIR, DATASET, name])
    _data = pd.read_csv(_path)
    _last_frame = _data["frame_id"].max() + 1
    ped_data_for_RA = {}
    pedestrian_data = {}.fromkeys(list(range(0,_last_frame)))
    [pedestrian_data.update({i:{}}) for i in pedestrian_data.keys()]
    for _id in _data["track_id"].unique():
        ped = _data.loc[_data["track_id"] == _id]
        _, _f, x, y, vx, vy, ax, ay = ped["track_id"], ped["frame_id"], ped["x"], ped["y"], ped["vx"], ped["vy"], ped["ax"], ped["ay"]
        ped_data_for_RA.update({_id: {"frame_id": _f, "x": x, "y": y, "vx": vx, "vy": vy, "ax": ax, "ay": ay}}) 
    _data_chunks_for_RA = generate_input_for_sim(ped_data_for_RA, dnn, _last_frame, input_len, load_data)
    for _det in _data.values:
        _id, _f, _, _, x, y, vx, vy, ax, ay = _det
        if _id != "P2": # Specific for 8_02_1
            pedestrian_data[_f].update({_id: {"x": x, "y": y, "vx": vx, "vy": vy, "ax": ax, "ay": ay}})
    return pedestrian_data, _data_chunks_for_RA, _last_frame

def generate_input_for_sim(data: dict, dnn: DNN, _last_frame: int, input_len: int = 90, load_data: bool = False):
    """ Generate the trajectory chunks for reachability analysis

        Parameters:
        -----------
        data : dict
            Dictionary of pedestrian data
        _last_frame : int
            The last frame in the dataset
        input_len : int
            The length of each chunk
    """
    if not load_data:
        _concat_data = {}.fromkeys(list(range(0,_last_frame)))
        [_concat_data.update({i:{}}) for i in _concat_data.keys()]
        for _j, _data in tqdm(data.items(), desc="Retreiving input"):
            if _j != "P2": # Specific for 8_02_1
                _f, x, y, vx, vy, ax, ay = _data["frame_id"], _data["x"], _data["y"], _data["vx"], _data["vy"], _data["ax"], _data["ay"]
                for _i in range(input_len, len(x)):
                    _x, _y = np.array(x.iloc[_i-input_len:_i]), np.array(y.iloc[_i-input_len:_i])
                    _vx, _vy = np.array(vx.iloc[_i-input_len:_i]), np.array(vy.iloc[_i-input_len:_i])
                    _ax, _ay = np.array(ax.iloc[_i-input_len:_i]), np.array(ay.iloc[_i-input_len:_i])
                    _frame = _f.values[_i]
                    _concat_data[_frame].update({_j: {"x": _x, "y": _y, "vx": _vx, "vy": _vy, "ax": _ax, "ay": _ay}})
        return _concat_data
        _classes = {}.fromkeys(LABELS.values())
        [_classes.update({i:[]}) for i in _classes.keys()]
        _new_data = {}.fromkeys(list(range(0,_last_frame)))
        [_new_data.update({i:{}}) for i in _new_data.keys()]
        for _frame, _data in tqdm(_concat_data.items(), desc="Getting classes for trajectories"):
            if len(_data.keys()) > 0:
                _x_arr = np.array([])
                for _i, _chunk in _data.items():
                    x, y, vx, vy, ax, ay = _chunk.values()
                    _x = np.hstack((x, y, vx, vy, ax, ay))
                    _x_arr = np.vstack((_x_arr, _x)) if _x_arr.size else _x
                c = dnn.predict(_x_arr)
                c = np.argmax(c, axis=1)
                for _c in c:
                    _classes[_c].append([_frame, _i])
        _lens = [len(v) for v in _classes.values()]
        _min_len = min(_lens)
        for _v in tqdm(_classes.values(), desc="Cleaning up chunks"):
            _ids = np.random.randint(0, len(_v), size=_min_len)
            for _id in _ids:
                _f, _i = _v[_id]
                _append = {_i: _concat_data[_f][_i]}
                _new_data[_f].update(_append)
        _file = open(ROOT + "/sim_dict.json", "wb")
        pickle.dump(_new_data, _file)
        _file.close()
    else:
        _file = open(ROOT + "/sim_dict.json", "rb")
        _new_data = pickle.load(_file)
        _file.close()
    return _new_data

def generate_args(d: np.ndarray, z: pp.zonotope, v: np.ndarray, c: int, input_len: int = 90, _N: int = 30, drop_equal: bool = True):
    """ Generate the necessary arguments for LTI reachability

        Parameters:
        -----------
        d : np.ndarray
        z : pp.zonotope
        v : np.ndarray
        c : int
        input_len : int (default = 30)
        _N : int (default = 90)
        drop_equal : bool (default = True)
    """
    U, X_p, X_m, _ = create_io_state(d, z, v, c, input_len=input_len, drop_equal=drop_equal)
    if not U.size: U, X_p, X_m, _ = create_io_state(d, z, v, list(LABELS.values()), input_len=input_len, drop_equal=drop_equal)
    _, _, U_traj = split_io_to_trajs(X_p, X_m, U, threshold=5, dropped=drop_equal, N=(_N-1))
    U_k = input_zonotope(U_traj, N=(_N-1))
    return U, X_p, X_m, U_k

def calculate_reachability(data: dict, d: np.ndarray, dnn: DNN, _last_frame: int, _N: int = 30, input_len: int = 90, 
                           drop_equal: bool = True, process_noise: float = 0.005, use_multiprocessing: bool = False):
    """ Calculate the LTI reachability for pedestrian in each frame

        Parameters:
        -----------
        data : dict
        d : np.ndarray
        dnn : DNN
        _last_frame : int
        _N : int (default = 90)
        input_len : int (default = 30)
        drop_equal : bool (default = True)
        process_noise : float (default = 0.005)
        use_multiprocessing : bool (default = False)
    """
    _RA = {}.fromkeys(list(range(0,_last_frame)))
    [_RA.update({i:[]}) for i in _RA.keys()]
    z_w = zonotope(np.array([0,0]), process_noise*np.ones(shape=(2,1)))
    _G = np.array([[4,0,2],[0,3,1]])
    G_z = np.array([[0.5,0,0.25],[0,0.5,0.15]])
    if use_multiprocessing:
        p = Pool(processes = multiprocessing.cpu_count())
    for _frame, _data in tqdm(list(data.items())[0:_last_frame], desc="Calculating reachable sets for frame"):
        if use_multiprocessing:
            t = time.time()
            _x_arr = np.array([])
            for _, _chunk in _data.items():
                x, y, vx, vy, ax, ay = _chunk.values()
                _arr = np.hstack((x, y, vx, vy, ax, ay))
                _x_arr = np.vstack((_x_arr, _arr)) if _x_arr.size else _arr
            if len(_x_arr) > 0:
                c = dnn.predict(_x_arr)
                c = np.argmax(c, axis=1)
                R = chunk_reachability(p, _data, d, c, drop_equal, input_len, _N, process_noise)
            else:
                R = []
            print(" ", R, time.time()-t)
            _RA[_frame] = R
        else:
            t = time.time()
            for _, _chunk in _data.items():
                x, y, vx, vy, ax, ay = _chunk.values()
                _arr = np.hstack((x, y, vx, vy, ax, ay))
                _x = np.array([x[-1], y[-1]])
                v = np.array([vx[-1], vy[-1]])
                z_oversize = zonotope(c_z=_x, G_z=_G)
                # TODO: pre-predict all modes maybe
                c = dnn.predict(_arr)
                c = np.argmax(c)
                U, X_p, X_m, U_k = generate_args(d, z_oversize, v, c, input_len, _N, drop_equal)
                M_w = create_M_w(U.shape[1], z_w, disable_progress_bar=True)
                z = zonotope(_x, G_z)
                R_arr = LTI_reachability(U, X_p, X_m, z, z_w, M_w, U_k, N=(_N-1), disable_progress_bar=True)
                R = R_arr[-1]
                R.color = [0,0.6,0]
                _RA[_frame].append(R)
            print(time.time()-t)
    if use_multiprocessing:
        p.close()
    _f = open(ROOT + RA_PATH, "wb")
    pickle.dump(_RA, _f)
    return _RA

def func(_chunk: dict, d: np.ndarray, c: int, drop_equal: bool, input_len: int, _N: int, process_noise: float):
    """ Functions that calculates the reachable sets for all pedestrians in one 
        frame (used in multiprocessing only)

        Parameters:
        ----------
        _chunk : dict
        d : np.ndarray
        c : int
        drop_equal : bool
        input_len : int
        _N : int
        process_noise : float
    """
    x, y, vx, vy, ax, ay = _chunk.values()
    _arr = np.hstack((x,y,vx,vy,ax,ay))
    _x = np.array([x[-1], y[-1]])
    _G = np.array([[4,0,2],[0,3,1]])
    v = np.array([vx[-1], vy[-1]])
    z_oversize = zonotope(c_z=_x, G_z=_G)
    U, X_p, X_m, U_k = generate_args(d, z_oversize, v, c, input_len, _N, drop_equal)
    z_w = zonotope(np.array([0,0]), process_noise*np.ones(shape=(2,1)))
    M_w = create_M_w(U.shape[1], z_w, disable_progress_bar=True)
    G_z = np.array([[0.5,0,0.25],[0,0.5,0.15]])
    z = zonotope(_x, G_z)
    R_arr = LTI_reachability(U, X_p, X_m, z, z_w, M_w, U_k, N=(_N-1), disable_progress_bar=True)
    R = R_arr[-1]
    return R

def chunk_reachability(p, _data: dict, d: np.ndarray, c: np.ndarray, drop_equal: bool, input_len: int, _N: int, process_noise: float):
    """ Chunk reachability used for multiprocessing the reachability analysis calculations

        Parameters:
        -----------
        _data : dict
        d : np.ndarray
        c : np.ndarray
        drop_equal : bool
        input_len : int
        _N : int
        process_noise : float
    """
    _arg = [_c for _, _c in _data.items()]
    _args = zip(_arg, repeat(d), c, repeat(drop_equal), repeat(input_len), repeat(_N), repeat(process_noise))
    res = p.starmap(func, _args)
    return res

def train_dnn(input_len: int = 90, load_dnn: bool = False):
    """ Train the classifier on the data

        Parameters:
        -----------
        input_len : int (default = 30)
        load_dnn : bool (default = False)
    """
    _sind = SinD()
    if input_len == 90:#input("Load? (y/n)") == "y":
        data = load_data()
        labels = load_data("sind_labels.pkl")
        train_data, _, train_labels, _ = split_data(data, labels)
    else:
        data = _sind.data(input_len=input_len)
        labels = _sind.labels(data, input_len=input_len)
        train_data, _, train_labels, _ = split_data(data, labels)
    train_data, train_labels = structure_input_data(train_data, train_labels)
    d = separate_data_to_class(train_data, train_labels)
    inp_size, out_size = train_data.shape[1], len(LABELS.keys())
    dnn = DNN(input_size=inp_size, output_size=out_size, nodes=[300, 150])
    if not load_dnn:
        dnn.train(train_data, train_labels, epochs=10)
    else:
        dnn = __load_dnn(dnn)
    return d, dnn

def __load_dnn(dnn: DNN) -> DNN:
    """ Loads the previously saved weights into a DNN

        Parameters:
        -----------
        dnn : DNN
    """
    return None # TODO: REMOVE
    dnn.load()
    return dnn

def __load_RA() -> dict:
    """ Loads the reachable sets """
    _f = open(ROOT + RA_PATH, "rb")
    return pickle.load(_f)


def simulate(map: SinD_map, drop_equal: bool = True, input_len: int = 30, _N: int = 90, frames: int = None, 
             load_dnn: bool = False, load_RA: bool = False, frequency: float = 10, use_multiprocessing: bool = False, 
             load_data: bool = False, plot_future_locs: bool = True):
    """ Simulate a dataset with visualization of pedestrians
        and their reachable set

        Parameters:
        -----------
        map : SinD_map
        drop_equal : bool (default = True)
        input_len : int (default = 30)
        _N : int (default = 90)
        frames : int (default = None)
        load_dnn : bool (default = False)
        load_RA : bool (default = False)
        frequency : float (default = 10)
        use_multiprocessing : bool (default = False)
    """
    d, dnn = train_dnn(input_len, load_dnn) if not load_RA else train_dnn(input_len, True)
    _data, _RA_data, _last_frame = load_data_for_simulation(dnn, input_len=input_len, load_data=load_data)
    _last_frame = _last_frame if not frames else frames
    _sets = calculate_reachability(_RA_data, d, dnn, _last_frame, _N, input_len, drop_equal, use_multiprocessing=use_multiprocessing) if not load_RA else __load_RA()
    plt.ion()
    ax, fig = map.plot_areas()
    images = []
    for frame in _data.keys() if not frames else list(_data.keys())[170:frames]:
        _t = time.time()
        x, y = [], []
        ids = []
        sc = ax.scatter(x, y, c="r", s=30, marker="o")
        for _id, p in _data[frame].items():
            x.append(p["x"]), y.append(p["y"])
            ids.append(_id)
        f_x, f_y = [], []
        for _id in ids:
            try:
                f_x.append(_data[frame+_N][_id]["x"])
                f_y.append(_data[frame+_N][_id]["y"])
            except Exception:
                continue
        sc.set_offsets(np.c_[x,y])
        visualize_zonotopes(_sets[frame], ax, plot_vertices=False)
        if plot_future_locs:
            f_sc = ax.scatter(f_x, f_y, s=30, c="b", marker="x")
            f_sc.set_offsets(np.c_[f_x,f_y]) if plot_future_locs else None
        fig.canvas.draw_idle()
        plt.pause(0.001)
        sc.remove()
        f_sc.remove() if plot_future_locs else None
        ax.get_children()[-11].remove()
        _remaining = 1/frequency - (time.time() - _t) if frequency is not None else 0
        time.sleep(_remaining) if _remaining > 0 else None
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
    imageio.mimsave(ROOT+"/SinD/plot.gif", images)



if __name__ == "__main__":
    map = SinD_map()
    input_len, _N = 90, 30
    t = time.time()
    simulate(map, True, input_len=input_len, _N=_N, frames=300, load_dnn=False, load_RA=True, load_data=False)
    print(time.time()-t)