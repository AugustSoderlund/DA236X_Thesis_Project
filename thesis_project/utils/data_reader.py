from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    from map import SinD_map
else:
    from .map import SinD_map
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
        
        generate_trajectories(cov: int, n: int) -> list[trajectories]
            generates n new trajectories sampled from the sets that are
            produced from the measurement and its uncertainty

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
        self.pedestrian_data = {}
        for dataset in self._DATASETS:
            _path = "/".join([ROOT, self._DATADIR, dataset, name])
            _data = pd.read_csv(_path)
            for _id in _data["track_id"].unique():
                ped = _data.loc[_data["track_id"] == _id]
                x, y, vx, vy, ax, ay = ped["x"], ped["y"], ped["vx"], ped["vy"], ped["ax"], ped["ay"]
                self.pedestrian_data.update({i: {"x": x, "y": y, "vx": vx, "vy": vy, "ax": ax, "ay": ay}}) 
                i += 1  


    def generate_trajectories(self, cov: int = 1, n: int = 2):
        # Generate new trajectories given uncertainty
        # Should be done before the self.data() function
        pass

    def data(self, input_len: int = 30, save_data: bool = True):
        _concat_data = []
        for _data in tqdm(self.pedestrian_data.values()):
            _ped_data = []
            x, y, vx, vy, ax, ay = _data["x"], _data["y"], _data["vx"], _data["vy"], _data["ax"], _data["ay"]
            for _i in range(input_len, len(_data["x"])):
                _to_append = np.array([*x.iloc[_i-input_len:_i], *y.iloc[_i-input_len:_i], 
                                       *vx.iloc[_i-input_len:_i], *vy.iloc[_i-input_len:_i], 
                                       *ax.iloc[_i-input_len:_i], *ay.iloc[_i-input_len:_i]])
                _ped_data.append(_to_append)
            _concat_data = [*_concat_data, *_ped_data] if len(_ped_data) > 0 else _concat_data
        if save_data:
            _f = open(ROOT + "/sind.pkl", "wb")
            pickle.dump(np.array(_concat_data), _f)
        return np.array(_concat_data)
        

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
