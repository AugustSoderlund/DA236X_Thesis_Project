import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from shapely.geometry import LineString

if __package__ or "." in __name__:
    from .map import SinD_map
else:
    from map import SinD_map
from tqdm import tqdm
import pickle

"""
This script includes classes that read data from different datasets and puts the data 
into a dictionary of the same format.
dict =  {
            _id:    {
                        "x": list[float],
                        "y": list[float],
                        "vx": list[float],
                        "vy": list[float],
                        "ax": list[float],
                        "ay": list[float]
                    }
        }

"""


ROOT = os.getcwd() + "/thesis_project/.datasets"
LABELS = {"crossing": 0,
          "not_crossing": 1,
          "crossing_illegally": 2,
          "crossing_cautious": 3,
          "unknown": 4}


class SinD:
    """ Class that reads the data from SinD dataset.

        Parameters:
        -----------
        name : str
            The name of the file from the SinD dataset that will be read 
            and used (default: 'Ped_smoothed_tracks')

        file_extension : str
            The extension of the file (default: '.csv')


        Functions:
        -----------
        data(input_len: int, save_data: bool) -> np.array
            retrieves every input_len part of the trajectories and
            returns a numpy array containing the data inside

        labels(data: np.ndarray, input_len: int) -> np.ndarray
            labels all the data such that the ground truth is known, 
            this assumes that certain areas/polygons in the map also
            is known

        plot_dataset() -> None
            plots both a 2D plot of the historical locations along the
            trajectory, a 3D plot containing the velocity profile, and
            a 3D plot for the acceleration profile
    """

    def __init__(self, name: str = "Ped_smoothed_tracks", file_extension: str = ".csv"):
        self._DATADIR = "SinD/Data"
        self._DATASETS = os.listdir("/".join([ROOT, self._DATADIR]))
        self._DATASETS.pop(self._DATASETS.index("mapfile-Tianjin.osm"))
        self.map = SinD_map()
        self.__load_dataset(name+file_extension)

    def __load_dataset(self, name):
        i = 0
        self.frequency = 1 / (100.100100100 / 1000)
        self.pedestrian_data = {}
        for dataset in self._DATASETS:
            _path = "/".join([ROOT, self._DATADIR, dataset, name])
            _data = pd.read_csv(_path)
            for _id in _data["track_id"].unique():
                ped = _data.loc[_data["track_id"] == _id]
                x, y, vx, vy, ax, ay = ped["x"], ped["y"], ped["vx"], ped["vy"], ped["ax"], ped["ay"]
                self.pedestrian_data.update({i: {"x": x, "y": y, "vx": vx, "vy": vy, "ax": ax, "ay": ay}}) 
                i += 1  

    def data(self, input_len: int = 30, save_data: bool = True):
        _concat_data = []
        for _data in tqdm(self.pedestrian_data.values(), desc="Retreiving input"):
            _ped_data = []
            x, y, vx, vy, ax, ay = _data["x"], _data["y"], _data["vx"], _data["vy"], _data["ax"], _data["ay"]
            for _i in range(input_len, len(_data["x"])):
                _extracted_data = np.array([*x.iloc[_i-input_len:_i], *y.iloc[_i-input_len:_i], 
                                       *vx.iloc[_i-input_len:_i], *vy.iloc[_i-input_len:_i], 
                                       *ax.iloc[_i-input_len:_i], *ay.iloc[_i-input_len:_i]])
                _ped_data.append(_extracted_data)
            _concat_data = [*_concat_data, *_ped_data] if len(_ped_data) > 0 else _concat_data
        if save_data:
            _f = open(ROOT + "/sind.pkl", "wb")
            pickle.dump(np.array(_concat_data), _f)
        return np.array(_concat_data)
    
    def labels(self, data: np.ndarray, input_len: int = 30):
        _labels = []
        for _data in tqdm(data, desc="Labeling data"):
            _x, _y = _data[0:input_len], _data[input_len:2*input_len] 
            _l = LineString(list(zip(_x, _y)))
            if (_l.within(self.map.road_poly) or _l.within(self.map.intersection_poly) or _l.within(self.map.gap_poly)) and not _l.within(self.map.crosswalk_poly):
                _labels.append(LABELS["crossing_illegally"])
            elif _l.within(self.map.crosswalk_poly) and not (_l.within(self.map.road_poly) or _l.within(self.map.intersection_poly) or _l.within(self.map.gap_poly)):
                _labels.append(LABELS["crossing"])
            elif _l.within(self.map.sidewalk_poly) and not (_l.within(self.map.road_poly) or _l.within(self.map.intersection_poly) or _l.within(self.map.gap_poly) or _l.within(self.map.crosswalk_poly)):
                _labels.append(LABELS["not_crossing"])
            elif _l.within(self.map.crosswalk_poly):
                _labels.append(LABELS["crossing_cautious"])
            elif _l.within(self.map.road_poly) or _l.within(self.map.intersection_poly):
                _labels.append(LABELS["crossing_illegally"])
            elif _l.within(self.map.sidewalk_poly):
                _labels.append(LABELS["not_crossing"])
            else:
                _labels.append(LABELS["unknown"])
        return np.array(_labels)
            


        

    def plot_dataset(self, map_overlay: bool = True, alpha: float = 0.2):
        ax1 = plt.figure(1).add_subplot(projection='3d')
        ax2 = self.map.plot_areas(alpha=alpha) if map_overlay == True else plt.figure(2).add_subplot()
        ax3 = plt.figure(3).add_subplot(projection='3d')
        for _id in self.pedestrian_data.keys():
            x, y = self.pedestrian_data[_id]["x"], self.pedestrian_data[_id]["y"]
            vx, vy = self.pedestrian_data[_id]["vx"], self.pedestrian_data[_id]["vy"]
            ax, ay = self.pedestrian_data[_id]["ax"], self.pedestrian_data[_id]["ay"]
            v = np.sqrt(np.array(vx).T**2+np.array(vy).T**2)
            a = np.sqrt(np.array(ax).T**2+np.array(ay).T**2)
            ax1.plot(x, y, zs=v, c="r"), ax1.set_title("Velocity profile of trajectories")
            ax2.plot(x, y, c="orange"), ax2.set_title("Pedestrian trajectories")
            ax3.plot(x, y, zs=a, c="r"), ax3.set_title("Acceleration profile of trajectories")
        plt.grid()
        plt.show()



class inD:
    def __init__(self, name: str = "Ped_smoothed_tracks", file_extension: str = ".csv"):
        self._DATADIR = "SinD/Data"
        self._DATASETS = os.listdir("/".join([ROOT, self._DATADIR]))
        self._DATASETS.pop(self._DATASETS.index("mapfile-Tianjin.osm"))
        #self.map = inD_map()
        self.__load_dataset(name+file_extension)



if __name__ == "__main__":
    data = SinD()
    d = data.data()
    print(d)
