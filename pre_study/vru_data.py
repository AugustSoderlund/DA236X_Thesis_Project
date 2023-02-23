import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point


ROOT = "C:/Users/ASOWXU/Desktop/Thesis Project/Code/OpenTraj/datasets"
DATASET = "/VRU/pedestrians"


class VRU_Data:
    def __init__(self):
        self.__initialize()

    def __load_dataset(self):
        traj, j = {}, 1
        crossing_peds = {}
        for folder in os.listdir(ROOT+DATASET)[1:2]:
            for file in os.listdir(ROOT+DATASET+"/"+folder):
                data = pd.read_csv(ROOT+DATASET+"/"+folder+"/"+file)
                x, y = [*data["x"].values.T], [*data["y"].values.T]
                for k in range(0,len(x),round(len(x)/20)):
                    p = Point((x[k], y[k]))
                    if not p.within(self.boundaries):
                        crossing_peds.update({j:j})
                        break
                traj.update({j: {"x":x, "y":y}})
                j += 1
        return traj, crossing_peds

    def plot_data(self):
        for id in self.data.keys():
            color = "r" if id in self.cross_ids else "b"
            data = self.data[id]
            x, y = data["x"], data["y"]
            if len(x) > 100:
                plt.scatter(x,y, s=0.05, c=color)
        plt.plot(self.boundary_plot["x"], self.boundary_plot["y"], 'g-', linewidth=4)
        plt.grid()
        plt.show()

    def __boundary(self):
        x, y = [-0.17, 1.32, -3, -5.35, -5.35], [-7.03, -0.58, 4.6, 4.75, -7.5]
        points = []
        for p in zip(x,y):
            points.append(Point(p))
        polygon = Polygon(points)
        self.boundaries = polygon
        self.boundary_plot = {"x":x[0:3], "y":y[0:3]}

    def __initialize(self):
        self.__boundary()
        self.data, self.cross_ids = self.__load_dataset()
        self.size = len(self.data.keys())


if __name__ == "__main__":
    data = VRU_Data()
    data.plot_data()