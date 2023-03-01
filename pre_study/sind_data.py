import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np


ROOT = "C:/Users/ASOWXU/Desktop/Thesis Project/Code"
DATASET = "/SinD/Data/8_02_1/"


class SIND_Data:
    def __init__(self, name="Ped_smoothed_tracks", file_extension=".csv"):
        self.__load_dataset(name+file_extension)

    def __load_dataset(self, name):
        _path = ROOT + DATASET + name
        _data = pd.read_csv(_path)
        self.pedestrian_data = {}
        for _id in _data["track_id"].unique():
            ped = _data.loc[_data["track_id"] == _id]
            x, y, vx, vy, ax, ay = ped["x"], ped["y"], ped["vx"], ped["vy"], ped["ax"], ped["ay"]
            self.pedestrian_data.update(
                {_id: {"x": x, "y": y, "vx": vx, "vy": vy, "ax": ax, "ay": ay}})

    def plot_dataset(self):
        ax1 = plt.figure(1).add_subplot(projection='3d')
        ax2 = plt.figure(2).add_subplot()
        for _id in self.pedestrian_data.keys():
            x, y = self.pedestrian_data[_id]["x"], self.pedestrian_data[_id]["y"]
            vx, vy = self.pedestrian_data[_id]["vx"], self.pedestrian_data[_id]["vy"]
            v = np.sqrt(np.array(vx).T**2+np.array(vy).T**2)
            ax1.plot(x, y, zs=v, c="r")
            ax2.plot(x, y, c="r")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    data = SIND_Data()
    data.plot_dataset()
