from itertools import repeat
from typing import List, Tuple, Union
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import pickle
from tqdm import tqdm
import pypolycontain as pp
import multiprocessing
from multiprocessing import Pool, Process
import imageio
import math
from copy import deepcopy
from shapely.geometry import Point, Polygon



if __package__ or "." in __name__:
    from utils.map import SinD_map
    from utils.data_reader import SinD, LABELS
    from PTC.dnn_classifier import DNN
    from DRA.operations import visualize_zonotopes, input_zonotope, create_M_w, optimize_vertices, is_inside
    from DRA.reachability import LTI_reachability
    from PTC.input_state import create_io_state, separate_data_to_class, split_io_to_trajs
    from DRA.zonotope import zonotope
    from utils.data_processor import load_data, structure_input_data, split_data
    from utils.evaluation import zonotope_area
    from utils.visualization import visualize_class
else:
    from map import SinD_map
    from ..DRA.operations import visualize_zonotopes
    from ..DRA.reachability import LTI_reachability

ROOT = os.getcwd() + "/thesis_project/.datasets"
DATADIR = "SinD/Data"
DATASET = "8_02_1"
RA_PATH = "/SinD/reachable_sets.pkl"
RAB_PATH = "/SinD/reachable_base_sets.pkl"

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
                for _i in range(0, len(x)-input_len):
                    _x, _y = np.array(x.iloc[_i:_i+input_len]), np.array(y.iloc[_i:_i+input_len])
                    _vx, _vy = np.array(vx.iloc[_i:_i+input_len]), np.array(vy.iloc[_i:_i+input_len])
                    _ax, _ay = np.array(ax.iloc[_i:_i+input_len]), np.array(ay.iloc[_i:_i+input_len])
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

def generate_args(d: np.ndarray, z: pp.zonotope, v: np.ndarray, c: Union[int, list], input_len: int = 90, _N: int = 30, drop_equal: bool = True):
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
                           drop_equal: bool = True, process_noise: float = 0.005, use_multiprocessing: bool = False, 
                           _baseline: bool = False, save_checkpoint: float = None):
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
    _RA_b = deepcopy(_RA)
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
            for _id, _chunk in _data.items():
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
                _RA[_frame].append((_id, R))
                if _baseline:
                    U_b, X_p_b, X_m_b, U_k_b = generate_args(d, z_oversize, v, list(LABELS.values()), input_len, _N, drop_equal)
                    M_w_b = create_M_w(U_b.shape[1], z_w, disable_progress_bar=True)
                    R_b_arr = LTI_reachability(U_b, X_p_b, X_m_b, z, z_w, M_w_b, U_k_b, N=(_N-1), disable_progress_bar=True)
                    R_b = R_b_arr[-1]
                    R_b.color = [0.55,0.14,0.14]
                    _RA_b[_frame].append((_id, R_b))
            print(time.time()-t)
        if save_checkpoint and _frame % save_checkpoint == 0:
            _f = open(ROOT + RA_PATH, "wb")
            pickle.dump(_RA, _f)
            _f.close()
            if _baseline:
                _f2 = open(ROOT + RAB_PATH, "wb")
                pickle.dump(_RA_b, _f2)
                _f2.close()
    if use_multiprocessing:
        p.close()
    _f = open(ROOT + RA_PATH, "wb")
    pickle.dump(_RA, _f)
    _f.close()
    if _baseline:
        _f2 = open(ROOT + RAB_PATH, "wb")
        pickle.dump(_RA_b, _f2)
        _f2.close()
    return _RA, _RA_b

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
        dnn.train(train_data, train_labels, epochs=300)
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

def __load_RA() -> Tuple[dict, dict]:
    """ Loads the reachable sets """
    _f = open(ROOT + RA_PATH, "rb")
    _f2 = open(ROOT + RAB_PATH, "rb")
    return pickle.load(_f), pickle.load(_f2)


def simulate(map: SinD_map, drop_equal: bool = True, input_len: int = 30, _N: int = 90, frames: int = None, 
             load_dnn: bool = False, load_RA: bool = False, frequency: float = 10, use_multiprocessing: bool = False, 
             load_data: bool = False, plot_future_locs: bool = True, calc_baseline: bool = True, 
             calc_areas: bool = True, calc_accuracy: bool = True):
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
    _modal_areas, _base_areas = [], []
    _modal_acc, _base_acc = [], []
    d, dnn = train_dnn(input_len, load_dnn) if not load_RA else train_dnn(input_len, True)
    _data, _RA_data, _last_frame = load_data_for_simulation(dnn, input_len=input_len, load_data=load_data)
    _last_frame = _last_frame if not frames else frames
    _sets, _baseline_sets = calculate_reachability(_RA_data, d, dnn, _last_frame, _N, input_len, drop_equal, use_multiprocessing=use_multiprocessing, _baseline=calc_baseline, save_checkpoint=50) if not load_RA else __load_RA()
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
        #_modal_id, _sets_ = list(zip(*_sets[frame])) if len(_sets[frame]) > 0 else [], []
        #_base_id, _baseline_sets_ = list(zip(*_baseline_sets[frame])) if len(_baseline_sets[frame]) > 0 else [], []
        _modal = list(zip(*_sets[frame]))# if len(_sets[frame]) > 0 else []
        _baseline = list(zip(*_baseline_sets[frame]))# if len(_baseline_sets[frame]) > 0 else []
        if len(_modal) > 0: _modal_id, _sets_ = _modal[0], _modal[1]
        else:  _modal_id, _sets_ = [], []
        if len(_baseline) > 0:  _, _baseline_sets_ = _baseline[0], _baseline[1]
        else: _, _baseline_sets_ = [], []
        visualize_zonotopes([*_baseline_sets_, *_sets_], ax, plot_vertices=False) if calc_baseline else visualize_zonotopes(_sets_, ax, plot_vertices=False)
        if plot_future_locs:
            f_sc = ax.scatter(f_x, f_y, s=30, c="b", marker="x")
            f_sc.set_offsets(np.c_[f_x,f_y]) if plot_future_locs else None
        fig.canvas.draw_idle()
        plt.pause(0.001)
        sc.remove()
        f_sc.remove() if plot_future_locs else None
        ax.get_children()[-11].remove()
        if calc_areas:
            for i in range(len(_sets[frame])):
                _modal_areas.append(zonotope_area(_sets_[i]))
                _base_areas.append(zonotope_area(_baseline_sets_[i]))
        _remaining = 1/frequency - (time.time() - _t) if frequency is not None else 0
        time.sleep(_remaining) if _remaining > 0 else None
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        if calc_accuracy:
            for i,_id in enumerate(_modal_id):
                _i = ids.index(_id)
                try:
                    p = Point((f_x[_i], f_y[_i]))
                    _s_m, _s_b = Polygon(optimize_vertices(_sets_[i])), Polygon(optimize_vertices(_baseline_sets_[i]))
                    if p.within(_s_m): _modal_acc.append(1)
                    else: _modal_acc.append(0)
                    if p.within(_s_b): _base_acc.append(1)
                    else: _base_acc.append(0)
                except Exception:
                    continue
    imageio.mimsave(ROOT+"/SinD/plot.gif", images)
    if calc_areas:
        print("######### AREAS OF ZONOTOPES ##########")
        print("## Modal zonotope:       ", sum(_modal_areas)/len(_modal_areas))
        print("## Baseline zonotope:    ", sum(_base_areas)/len(_base_areas))
    if calc_accuracy:
        print("######### ACCURACY OF ZONOTOPES ##########")
        print("## Modal zonotope:       ", sum(_modal_acc)/len(_modal_acc))
        print("## Baseline zonotope:    ", sum(_base_acc)/len(_base_acc))


def reachability_for_specific_position_and_mode(pos: np.ndarray = np.array([-3.4,28.3]), c: int = 1, vel: np.ndarray = np.array([1,0]), 
                                                _baseline: bool = True, _show_plot: bool = True, _ax: plt.Axes = None, _labels: list = None,
                                                _suppress_prints: bool = False, _sind_: SinD = None, _d: np.ndarray = None, sim: bool = False):
    """ Get reachable set for a specific position, mode and starting velocity

        Parameters:
        -----------
        pos : np.ndarray
        c : int
        vel : np.ndarray
        _baseline : bool
        _show_plot : bool
        _ax : plt.Axes
        _labels : list
        _suppress_prints : bool
        _sind_ : SinD
        _d : np.ndarray
    """
    input_len = 90
    a = input_len-1
    if not _sind_: _sind = SinD()
    else: _sind = _sind_
    if type(_d) is not np.ndarray:
        if input_len == 90:
            data = load_data()
            labels = load_data("sind_labels.pkl")
            train_data, _, train_labels, _ = split_data(data, labels)
        else:
            data = _sind.data(input_len=input_len)
            labels = _sind.labels(data, input_len=input_len)
            train_data, _, train_labels, _ = split_data(data, labels)
        train_data, train_labels = structure_input_data(train_data, train_labels)
        d = separate_data_to_class(train_data, train_labels)
    else:
        d = _d
    c_z = pos
    G_z = np.array([[2,0,1],[0,2,0.6]])
    z = zonotope(c_z, G_z)
    v = vel
    classification = c
    # TODO: Change the classification below to c instead of [0,1,2,3,4,5,6] (used for only baseline computations)
    U, X_p, X_m, _ = create_io_state(d, z, v, [0,1,2,3,4,5,6], input_len=input_len, drop_equal=True, angle_filter=True)
    process_noise = 0.005
    _, _, U_traj = split_io_to_trajs(X_p, X_m, U, threshold=5, dropped=True, N=a)
    U_k = input_zonotope(U_traj, N=a)
    z_w = zonotope(np.array([0,0]), process_noise*np.ones(shape=(2,1)))
    M_w = create_M_w(U.shape[1], z_w, disable_progress_bar=sim)
    G_z = np.array([[0.5,0,0.25],[0,0.5,0.15]])
    z = zonotope(c_z, G_z)
    R = LTI_reachability(U, X_p, X_m, z, z_w, M_w, U_k, N=a, disable_progress_bar=sim)
    R_all = R
    R = R[-1]
    R.color = [0,0.6,0]
    R_base_all = None
    if _baseline:
        U_all, X_p_all, X_m_all, _ = create_io_state(d, z, v, [0,1,2,3,4,5,6], input_len=input_len, drop_equal=True, angle_filter=False)
        _, _, U_all_traj = split_io_to_trajs(X_p_all, X_m_all, U_all, threshold=5, dropped=True, N=a)
        U_k_all = input_zonotope(U_all_traj, N=a)
        M_w_base = create_M_w(U_all.shape[1], z_w, disable_progress_bar=sim)
        R_base = LTI_reachability(U_all, X_p_all, X_m_all, z, z_w, M_w_base, U_k_all, N=a, disable_progress_bar=sim)
        R_base_all = R_base
        R_base = R_base[-1]
        R_base.color = [0.55,0.14,0.14]
    if not _suppress_prints:
        print("Area of zonotope: ", round(zonotope_area(R), 4), " m^2")
        if _baseline: print("Area of (baseline) zonotope: ", round(zonotope_area(R_base), 4), " m^2")
    z = zonotope(c_z, G_z)
    if not _ax: _ax, _ = _sind.map.plot_areas()
    z.color = [1,1,55/255]
    _zonos = [R_base, R, z] if _baseline else [R, z]
    if sim: ax = visualize_zonotopes(_zonos, map=_ax, show=False, _labels=_labels)
    else: ax = None
    if _show_plot: plt.show()
    return ax, _zonos, R_all, R_base_all

def reachability_for_all_modes(pos: np.ndarray = np.array([-3.4,28.3]), vel: np.ndarray = np.array([1,0]), baseline: bool = False, _sind_: SinD = None, d_: np.ndarray = None, simulation: bool = False):
    """ Reachability for all modes

        Parameters:
        -----------
        pos : np.ndarray
        vel : np.ndarray
        _sind_ : SinD
        d_ : np.ndarray
    """
    if type(d_) is not np.ndarray: _sind_, d_ = calc_d()
    _colors = [[0.55,0.14,0.14], [0,0,0.8], [0,0.6,0], [1,0.5,0], [0,1,1], [1,0,1], [0.38,0.38,0.38]]
    _modes = list(LABELS.values())
    _labels =  ["Mode " + str(i) for i in _modes]
    _labels = list(LABELS.keys())
    ax, _ = _sind_.map.plot_areas()
    _z, _ids = [], []
    _b, _z_all = [], {}
    try:
        _, _zonos, R_all, _ = reachability_for_specific_position_and_mode(pos, _modes[0], vel, _baseline=baseline, _show_plot=False, _ax=ax, _suppress_prints=True, _sind_=_sind_, _d=d_, sim=simulation)
        if not baseline: R, z = _zonos
        else: R_base, R, z = _zonos; _b.append(R_base)
        R.color = _colors[0]
        _z.append(R)
        _ids.append(0)
        _z_all.update({_modes[0]: R_all})
    except Exception: pass
    for _mode in _modes[1:]:
        try:
            if baseline: baseline = True if not _b else False
            _, _zonos, R_all, _ = reachability_for_specific_position_and_mode(pos, _mode, vel, _baseline=baseline, _show_plot=False, _ax=ax, _suppress_prints=True, _sind_=_sind_, _d=d_, sim=simulation)
            if not baseline: R, z = _zonos
            else: R_base, R, z = _zonos; _b.append(R_base)
            R.color = _colors[_mode]
            _z.append(R)
            _ids.append(_mode)
            _z_all.update({_mode: R_all})
        except Exception: pass
    _l = [_labels[i] for i in _ids]
    #if not simulation:
    _z.append(z)
    _l.append("Initial set")
    _show = True
    #else:
    _show = False
    if _b: _l.insert(0, "Baseline"); _z.insert(0, _b[0])
    visualize_zonotopes(_z, map=ax, show=_show, _labels=_l)
    return _z, _l, _b, _z_all
   



def calc_d(_load: bool = False, drop_data: str = None, _sind: SinD = None):
    """ Calculate the data separation to each class """
    input_len = 90
    if not _sind: _sind_ = SinD(drop_file=drop_data)
    else: _sind_ = _sind
    if _load:
        data = load_data()
        labels = load_data("sind_labels.pkl")
        train_data, _, train_labels, _ = split_data(data, labels)
    else:
        data = _sind_.data(input_len=input_len)
        labels = _sind_.labels(data, input_len=input_len)
        train_data, _, train_labels, _ = split_data(data, labels)
    train_data, train_labels = structure_input_data(train_data, train_labels)
    d_ = separate_data_to_class(train_data, train_labels)
    return _sind_, d_

def generate_trajectory(f: float = 10.0):
    """ Generate a trajectory for simulation

        Parameters:
        -----------
        f : float (default = 10.0)
            Frequency of dataset
    """
    xy, v = [], []
    noise = 0.03
    avg_vel = 1.3
    _checkpoints = [(-3.4,28.3), (27,28.3), (50,3), (30,3), (27,28.3), (0,3), (-24,3)]
    for i in range(0,len(_checkpoints)-1):
        c1, c2 = np.array(_checkpoints[i]), np.array(_checkpoints[i+1])
        _norm = np.linalg.norm(c2-c1)
        num = math.ceil((_norm / avg_vel * f))
        _p = np.linspace(c1, c2, num)
        _noise = np.random.normal(scale=noise, size=_p.shape)
        _p = _p + _noise
        _vel = np.ones(shape=_p.shape) * ((c2-c1) / _norm * avg_vel) + _noise
        xy = [*xy, *_p]
        v = [*v, *_vel]
    return xy, v

def simulate_forged_traj(load: bool = False, _load_calc_d_data: bool = True):
    _sind, _d = calc_d(_load=_load_calc_d_data)
    xy, v = generate_trajectory()
    if not load:
        RA, RA_all = [], []
        _labels = []
        for i in tqdm(range(120), desc="Reachability analysis for all modes"):
            _z, _l, _, _z_all = reachability_for_all_modes(xy[i], v[i], _sind, _d, simulation=True)
            RA.append(_z); RA_all.append(_z_all)
            _labels.append(_l)
        _f = open(ROOT + "/SinD/all_sets_forged_traj.pkl", "wb")
        pickle.dump([RA, _labels, RA_all], _f)
        _f.close()
    else:
        _f = open(ROOT + "/SinD/all_sets_forged_traj.pkl", "rb")
        RA, _labels, RA_all = pickle.load(_f)
        _f.close()
    plt.ion()
    ax, fig = _sind.map.plot_areas()
    for i in range(120):
        visualize_zonotopes(RA[i], ax, plot_vertices=False, _labels=_labels[i])
        sc = ax.scatter(xy[i][0], xy[i][1], c="r", s=30, marker="o")
        fc = ax.scatter(xy[i+90][0], xy[i+90][1], c="b", s=30, marker="x")
        fig.canvas.draw_idle()
        plt.pause(0.001)
        sc.remove()
        fc.remove()
        ax.get_children()[-12].remove()


def _sim_func(sind: SinD, d_: np.ndarray, _data: dict, _RA_data: dict, RA: dict, RA_b: dict, input_len: int, checkpoint: int, frames: tuple, all_modes: bool, _baseline: bool):
    for frame in tqdm(_data.keys() if not frames else range(frames[0], frames[1]), desc="Simulating " + DATASET):
        _RA = _RA_data[frame]
        for _ped_id, state in _data[frame].items():
            if _ped_id in _RA:
                pos = np.array([state["x"], state["y"]])
                vel = np.array([state["vx"], state["vy"]])
                _chunk = np.array([np.hstack(_RA[_ped_id].values())])
                mode = sind.labels(_chunk, input_len=input_len, save_data=False, disable_progress_bar=True)[0]
                if all_modes:
                    _, _, _b, _z_all = reachability_for_all_modes(pos, vel, True, sind, d_=d_, simulation=True)
                    while mode not in _z_all:
                        sind, d_ = calc_d(_load = True, drop_data=DATASET, _sind=sind)
                        _, _, _b, _z_all = reachability_for_all_modes(pos, vel, True, sind, d_=d_, simulation=True)
                else: 
                    _,_, _z_all, _b_all = reachability_for_specific_position_and_mode(pos, mode, vel, _baseline, False, _suppress_prints=True, _sind_=sind, _d=d_, sim=True)
                    while not _z_all:
                        sind, d_ = calc_d(_load = True, drop_data=DATASET, _sind=sind)
                        _,_, _z_all, _b_all = reachability_for_specific_position_and_mode(pos, mode, vel, _baseline, False, _suppress_prints=True, _sind_=sind, _d=d_, sim=True)
                RA[frame].update({_ped_id: {"zonotopes": _z_all, "mode": mode}})
                if all_modes: RA_b[frame].update({_ped_id: _b})
        if frame % checkpoint and frame != 0:
            _f = open(ROOT + "/SinD/" + DATASET + ".pkl", "wb")
            pickle.dump([RA, RA_b, (frame, frames[1])], _f)
            _f.close()
    _f = open(ROOT + "/SinD/" + DATASET + ".pkl", "wb")
    pickle.dump([RA, RA_b, (frame, frames[1])], _f)
    _f.close()

def _simulation(input_len: int = 90, load_data: bool = True, load_calc_d_data: bool = True, checkpoint: int = 5, frames: int = None, all_modes: bool = False, continue_prev: bool = False, _baseline: bool = False):
    """ Simulating the DATASET using the "true" mode (from the labeling oracle)

        Parameters:
        -----------
        input_len : int (default = 90)
        load_data : bool (default = True)
        load_calc_d_data : bool (default = True)
        checkpoint : int (default = 10)
        frames : int (default = None)
        all_modes : bool (default = False)
    """
    sind, d_ = calc_d(_load = load_calc_d_data, drop_data=DATASET)
    _data, _RA_data, _last_frame = load_data_for_simulation(None, input_len=input_len, load_data=False)
    if continue_prev:
        _f = open(ROOT + "/SinD/" + DATASET + ".pkl", "rb")
        RA, RA_b, _frames = pickle.load(_f)
        _frames = _frames if not frames else (_frames[0], _frames[0]+frames)
        _f.close()
        _sim_func(sind, d_, _data, _RA_data, RA, RA_b, input_len, checkpoint, _frames, all_modes, _baseline=_baseline)
        load_data = True
    if not load_data:
        RA = {}.fromkeys(list(range(0,_last_frame)))
        [RA.update({i:{}}) for i in RA.keys()]
        RA_b = deepcopy(RA)
        for frame in tqdm(_data.keys() if not frames else range(0,frames), desc="Simulating " + DATASET):
            _RA = _RA_data[frame]
            for _ped_id, state in _data[frame].items():
                if _ped_id in _RA:
                    pos = np.array([state["x"], state["y"]])
                    vel = np.array([state["vx"], state["vy"]])
                    _chunk = np.array([np.hstack(_RA[_ped_id].values())])
                    mode = sind.labels(_chunk, input_len=input_len, save_data=False, disable_progress_bar=True)[0]
                    if all_modes:
                        _, _, _b, _z_all = reachability_for_all_modes(pos, vel, True, sind, d_=d_, simulation=True)
                        while mode not in _z_all:
                            sind, d_ = calc_d(_load = True, drop_data=DATASET, _sind=sind)
                            _, _, _b, _z_all = reachability_for_all_modes(pos, vel, True, sind, d_=d_, simulation=True)
                    else:
                        _z_all = None 
                        while not _z_all:
                            try:
                                sind, d_ = calc_d(_load = True, drop_data=DATASET, _sind=sind)
                                _,_, _z_all, _b_all = reachability_for_specific_position_and_mode(pos, mode, vel, _baseline, False, _suppress_prints=True, _sind_=sind, _d=d_, sim=True)
                            except Exception:
                                sind, d_ = calc_d(_load = True, drop_data=DATASET, _sind=sind)
                    RA[frame].update({_ped_id: {"zonotopes": _z_all, "mode": mode}})
                    if _baseline: RA_b[frame].update({_ped_id: _b_all})
            if frame % checkpoint and frame != 0:
                _f = open(ROOT + "/SinD/" + DATASET + ".pkl", "wb")
                pickle.dump([RA, RA_b, (frame, _last_frame)], _f)
                _f.close()
        _f = open(ROOT + "/SinD/" + DATASET + ".pkl", "wb")
        pickle.dump([RA, RA_b, (frame, _last_frame)], _f)
        _f.close()
    else:
        _f = open(ROOT + "/SinD/" + DATASET + ".pkl", "rb")
        RA, RA_b, _frames = pickle.load(_f)
        frames = _frames[1]
        _f.close()
    RA_acc = np.array([0] * input_len)
    i = np.array([0] * input_len)
    for frame in tqdm(RA.keys(), desc="Calculating accuracy for " + DATASET):
        _RA = _RA_data[frame]
        for _ped_id, state in _data[frame].items():
            if _ped_id in _RA and RA[frame]:
                _z_all, mode = RA[frame][_ped_id]["zonotopes"], RA[frame][_ped_id]["mode"]
                if _baseline: _b = RA_b[frame][_ped_id]
                for k in range(0,input_len):
                    state_k = _data[frame+k][_ped_id]
                    pos_k = np.array([state_k["x"], state_k["y"]])
                    try:
                        zono = _z_all[mode][k] if all_modes else _z_all[k]
                        #if _baseline: print(_b)
                        #ax, _ = sind.map.plot_areas()
                        #visualize_zonotopes([zono], ax, show=False)
                        #ax.scatter(pos_k[0], pos_k[1], s=10)
                        _inside = int(is_inside(zono, pos_k))
                        RA_acc[k] += _inside
                        i[k] += 1
                        #print(_inside)
                        #plt.show()
                    except Exception:
                        pass
    ids = np.where(i == 0)[0]
    RA_acc, i = RA_acc[0:ids[0]], i[0:ids[0]]
    print(RA_acc/i)
    _f = open(ROOT + "/state_inclusion_acc.pkl", "wb")
    pickle.dump(RA_acc/i*100, _f)
    _f.close()
    fig, _ = plt.subplots()
    fig.set_size_inches(5,2.4)
    plt.plot(list(range(1,len(RA_acc)+1)), RA_acc/i*100)
    plt.ylim([0,110])
    plt.xlim([0,90])
    plt.ylabel("Accuracy [%]")
    plt.xlabel("Time horizon, N [-]")
    plt.grid()
    plt.show()

def visualize_state_inclusion_acc(baseline: bool = True, convergence: bool = True, side: str = "right"):
    if baseline:
        _f = open(ROOT + "/state_inclusion_acc_modal_without_heading.pkl", "rb")
        _f_b = open(ROOT + "/state_inclusion_acc_baseline_without_heading.pkl", "rb")
    else:
        _f = open(ROOT + "/state_inclusion_acc.pkl", "rb")
    RA_acc = pickle.load(_f)
    RA_b_acc = pickle.load(_f_b)
    _f.close() 
    if baseline: _f_b.close()
    fig, ax = plt.subplots()
    fig.set_size_inches(10/1.5,4.2/1.5)
    fig.subplots_adjust(top=0.96, left=0.090, bottom=0.165, right=0.93)
    _x = np.array(list(range(1,len(RA_acc)+1)))/10
    ax.plot(_x, RA_acc, color="green")
    if baseline: plt.plot(_x, RA_b_acc, "--", color="red")
    ax.set_ylim([0,110]), ax.set_xlim([0,9])
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.minorticks_on()
    ax.set_ylabel("Accuracy [%]")
    ax.set_xlabel("Time horizon, N [s]")
    ax.legend(["Modal", "Baseline"]) if baseline else None
    ax.grid(which="major")
    ax.grid(which="minor", ls="--", linewidth=0.33)
    if convergence:
        ax2 = ax.twinx()
        ax2.set_yticks([91], ["91%"])
        ax2.tick_params(axis="y", colors="green", labelsize=10)
        ax2.grid(alpha=0.6)
        ax2.set_ylim([0,110])
        ax2.yaxis.set_ticks_position(side)
        ax3 = ax.twinx()
        ax3.set_yticks([98.7], ["98%"])
        ax3.tick_params(axis="y", colors="red", labelsize=10)
        ax3.grid(alpha=0.6)
        ax3.set_ylim([0,110])
        ax3.yaxis.set_ticks_position(side)

    #ax.set_yticks(yticks)
    #ylabels = ax.get_yticklabels()
    #ax.set_yticklabels(ylabels)

    #ax.arrow(9-3/10, RA_b_acc[-3], 0, RA_acc[-3]-RA_b_acc[-3], width=0.03)
    #ax.plot([9-3/10,9-3/10],[RA_b_acc[-3], RA_acc[-3]],"--",c="black", linewidth=2)
    #bbox = dict(boxstyle="round", fc="black", ec="black", alpha=0.2)
    #ax.annotate("Diff: -5 pp",
    #        xy=(9-3.4/10, RA_b_acc[-3]-2), xycoords="data", xytext =(-40,-60),
    #        textcoords ='offset points', bbox = bbox, color="black",
    #        arrowprops=dict(arrowstyle= '<|-|>',
    #                         color='black',
    #                         lw=3.5,
    #                         ls='-'))
    plt.show()


def format_volume_calc_for_latex():
    _, _, _, _, _z1_, _b1_ = load_data("scenario1.pkl")
    _, _, _, _, _z2_, _b2_ = load_data("scenario2.pkl")
    _, _, _, _, _z3_, _b3_ = load_data("scenario3.pkl")
    a = ["(a)", "(b)", "(c)"]
    _z = [_z1_, _z2_, _z3_]
    _b = [_b1_, _b2_, _b3_]
    for i in range(0,3):
        _l = [f"Scenario {a[i]}"]
        for mode in LABELS.values():
            modal_area = []
            try:
                for _zonotope in _z[i][mode]:
                    area = zonotope_area(_zonotope)
                    modal_area.append(area)
                #zono = _z[i][mode][-1]
                #area = zonotope_area(zono)
                #_l.append(f"{area:.{4}g}")
                modal_area = np.array(modal_area)
                _mean, _std = modal_area.mean(), modal_area.std()
                if str(_mean) == "nan" or str(_std) == "nan":
                    _l.append("---")
                else:
                    _l.append(f"${_mean:.{4}g} \pm {_std:.{4}g}$")
            except Exception:
                _l.append("---")
        baseline_area = []
        for _baseline_zono in _b[i]:
            _base_area = zonotope_area(_baseline_zono)
            baseline_area.append(_base_area)
        baseline_area = np.array([baseline_area])
        _mean, _std = baseline_area.mean(), baseline_area.std()
        _l.append(f"${_mean:.{4}g} \pm {_std:.{4}g}$")
        print(" & ".join(_l) + "\\\\")

def visualize_scenario(scenario: str = "1", all_modes: bool = True, save: bool = True, overlay_image: bool = True):
    func = "scenario" + scenario + ".pkl"
    heading = {"1":(5,0), "2": (5,0), "3": (-5,0)}
    arrows = {"1":r'$\rightarrow$', "2":r'$\rightarrow$', "3":r'$\leftarrow$'}
    _markers = np.array(["o", "s", "x", "p", "v", "_", "|", "*", "1", "h"])
    z, l, _z, *_ = load_data(func)
    if not all_modes:
        _b = z[0]
        _i = z[-1]
        _i.color = [0,0,0.8]
        for _mode, _zono in _z.items():
            _zono = _zono[-1]
            _zono.color = [0,0.6,0]
            _mode_str = list(LABELS.keys())[_mode]
            _title = _mode_str.replace("_", " ").capitalize() + " vs. Baseline"
            l = ["Baseline", _mode_str.replace("_", " ").capitalize(), "Initial set", "Initial heading"]
            markers = ["o", "s", "h"]
            if overlay_image:
                ax, _ = SinD_map().plot_areas()
                im = plt.imread(ROOT + "/intersection.jpg")
                ax.imshow(im, alpha=0.4, extent=[-25,60,-9.3,40])
                ax.set_xlim([-24,57])
                ax.set_ylim([-8.6,39.6])
            else:
                ax = SinD_map()
            ax.arrow(_i.x[0][0], _i.x[1][0], heading[scenario][0], heading[scenario][1], head_width=1, head_length=0.5, length_includes_head=True, color='black', zorder=10)
            ax.add_line((Line2D([10000], [10000], linestyle="none", marker=arrows[scenario], alpha=0.6, markersize=10, 
                                markerfacecolor="black", markeredgecolor="black", label="Initial heading")))
            visualize_zonotopes([_b, _zono, _i], ax, show=not save, _labels=l, _markers=markers, title=_title)
            plt.savefig(ROOT + "/scenario"+scenario+_mode_str+".png")
    else:
        ids = list(_z.keys())  
        ids.append(len(_markers)-2)
        ids.insert(0, len(_markers)-1)
        markers = _markers[ids]
        visualize_zonotopes(z, SinD_map(), show=not save, _labels=l, _markers=markers)
        plt.savefig(ROOT + "/scenario"+scenario+".png")

def scenario_func(pos: np.ndarray, vel: np.ndarray, scenario: int):
    _z_ = {}.fromkeys(LABELS.values())
    [_z_.update({i:[]}) for i in _z_.keys()]
    _b_ = []
    for _ in range(0,20):
        _sind, _d = calc_d(_load=True)
        z,l,_b,_z = reachability_for_all_modes(pos, vel, baseline=True, _sind_=_sind, d_=_d)
        while not _z:
            _sind, _d = calc_d(_load=True)
            z,l,_b,_z = reachability_for_all_modes(pos, vel, baseline=True, _sind_=_sind, d_=_d)
        for k,v in _z.items():
            if v:
                _z_[k].append(v[-1])
        _b_.append(_b[-1])
    _f = open(ROOT + f"/scenario{scenario}.pkl", "wb")
    pickle.dump([z, l, _z, _b, _z_, _b_], _f)
    _f.close()

def run_scenario(_s: int):
    scenario = f"scenario_{_s}()"
    eval(scenario)

def scenario_1():
    pos = np.array([-3.4, 28.3])
    vel = np.array([1,0])
    scenario_func(pos, vel, 1)

def scenario_2():
    pos = np.array([-6.5,4.20])
    #pos = np.array([26,0.5])
    vel = np.array([1,0])
    scenario_func(pos, vel, 2)

def scenario_3():
    pos = np.array([15,3])
    vel = np.array([-1,0])
    scenario_func(pos, vel, 3)

def vis_class_trajs(_class: int):
    sind = SinD()
    data = load_data()
    labels = load_data("sind_labels.pkl")
    visualize_class(sind.map, _class, data, labels, input_len=90)



def intersection_figure(name: str = "Ped_smoothed_tracks.csv", dataset: str = "8_02_1", ped_ids: list = ["P1", "P5", "P10"], modes: list = ["1", "2", "3"], use_len: bool = True, show: bool = False):
    if len(ped_ids) != len(modes):
        print("Lengthes of mode-list and id-list not equal!")
        modes = None
    bbox = dict(boxstyle="round", fc="white", ec="black", alpha=0.5)
    _map = SinD_map()
    ax, fig = _map.plot_areas()
    fig.set_size_inches(6.5, 3.87)
    fig.subplots_adjust(top=1, left=0, bottom=0, right=1)
    #im = plt.imread(ROOT + "/SinD/doc/Ground Coordinate System and Traffic Light Distribution.png")
    im = plt.imread(ROOT + "/intersection.jpg")
    ax.imshow(im, alpha=0.5, extent=[-25,60,-9.3,40])
    ax.set_xlim([-24,57])
    ax.set_ylim([-8.6,39.6])
    _path = "/".join([ROOT, DATADIR, dataset, name])
    _data = pd.read_csv(_path)
    first = True
    length = 10000000
    for i,_id in enumerate(ped_ids):
        _ped = _data.loc[_data["track_id"] == _id]
        _x, _y = _ped["x"].to_numpy(), _ped["y"].to_numpy()
        _center = round(len(_x)/2)
        if len(_x) < 60*2+1:
            length = round(len(_x)/2)-1
        _prev_traj_x, _prev_traj_y, _future_traj_x, _future_traj_y = _x[:_center], _y[:_center], _x[_center:], _y[_center:]
        if use_len:
            _prev_traj_x = _prev_traj_x[-length:]
            _prev_traj_y = _prev_traj_y[-length:]
            _future_traj_x = _future_traj_x[:length]
            _future_traj_y = _future_traj_y[:length]
        ax.plot(_prev_traj_x, _prev_traj_y, c="r", linewidth=2, label="Previous trajectory" if first else None)
        ax.plot(_future_traj_x, _future_traj_y, "--", c="g", linewidth=2, label="Future trajectory" if first else None)
        ax.scatter(_prev_traj_x[-1], _prev_traj_y[-1], s=50, c="blue", label="Pedestrian" if first else None)
        ax.annotate(modes[i] if modes else _id,
            (_prev_traj_x[-1], _prev_traj_y[-1]), xytext =(0, 12),
            textcoords ='offset points', bbox = bbox, color="black")
        if _id == ped_ids[0]:
            first = False
    plt.legend()
    plt.axis('off')
    if show: plt.show()
    plt.savefig(ROOT + "/og.png")





if __name__ == "__main__":
    map = SinD_map()
    input_len, _N = 90, 30
    t = time.time()
    simulate(map, True, input_len=input_len, _N=_N, frames=300, load_dnn=False, load_RA=True, load_data=False)
    print(time.time()-t)