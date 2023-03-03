import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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


ROOT = "C:/Users/ASOWXU/Desktop/Thesis Project/Code/DA236X_Thesis_Project/thesis_project/.datasets"


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
        generate_trajectories(cov: int, n: int)
            generates n new trajectories sampled from the sets that are
            produced from the measurement and its uncertainty

        plot_dataset()
            plots both a 2D plot of the historical locations along the
            trajectory, a 3D plot containing the velocity profile, and
            a 3D plot for the acceleration profile
    """

    def __init__(self, name: str = "Ped_smoothed_tracks", file_extension: str = ".csv"):
        self.__DATADIR = "SinD/Data"
        self.__DATASETS = os.listdir("/".join([ROOT, self.__DATADIR]))
        self.__map = self.__DATASETS.pop(self.__DATASETS.index("mapfile-Tianjin.osm"))
        self.__load_dataset(name+file_extension)

    def __load_dataset(self, name):
        i = 0
        self.pedestrian_data = {}
        for dataset in self.__DATASETS:
            _path = "/".join([ROOT, self.__DATADIR, dataset, name])
            _data = pd.read_csv(_path)
            for _id in _data["track_id"].unique():
                ped = _data.loc[_data["track_id"] == _id]
                x, y, vx, vy, ax, ay = ped["x"], ped["y"], ped["vx"], ped["vy"], ped["ax"], ped["ay"]
                self.pedestrian_data.update({i: {"x": x, "y": y, "vx": vx, "vy": vy, "ax": ax, "ay": ay}}) 
                i += 1  


    def generate_trajectories(self, cov: int = 1, n: int = 2):
        for _id in self.pedestrian_data.keys():
            pass

    def plot_dataset(self):
        ax1 = plt.figure(1).add_subplot(projection='3d')
        ax2 = plt.figure(2).add_subplot()
        ax3 = plt.figure(3).add_subplot(projection='3d')
        for _id in self.pedestrian_data.keys():
            x, y = self.pedestrian_data[_id]["x"], self.pedestrian_data[_id]["y"]
            vx, vy = self.pedestrian_data[_id]["vx"], self.pedestrian_data[_id]["vy"]
            ax, ay = self.pedestrian_data[_id]["ax"], self.pedestrian_data[_id]["ay"]
            v = np.sqrt(np.array(vx).T**2+np.array(vy).T**2)
            a = np.sqrt(np.array(ax).T**2+np.array(ay).T**2)
            ax1.plot(x, y, zs=v, c="r"), ax1.set_title("Velocity profile of trajectories")
            ax2.plot(x, y, c="r"), ax2.set_title("Pedestrian trajectories")
            ax3.plot(x, y, zs=a, c="r"), ax3.set_title("Acceleration profile of trajectories")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    data = SinD()
    data.plot_dataset()
