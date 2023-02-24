import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

ROOT = "C:/Users/ASOWXU/Desktop/Thesis Project/Code/OpenTraj/datasets"
DATASET = "/SDD/little/video0/annotations.txt"


class SDD_Data:
    def __init__(self):
        self.__load_dataset()
        self.__boundaries()

    def __load_dataset(self):
        pedestrians, i = {}, 0
        __file_dir = ROOT+DATASET
        f = open(__file_dir)
        id_prev = i
        for line in f.readlines():
            l = line.replace("\n", "").split(" ")
            _id = eval(l[0])
            if eval(l[-1]) == "Pedestrian":
                if id_prev != _id:
                    i += 1
                xmin, ymin, xmax, ymax = [eval(p) for p in l[1:5]]
                center = ((xmin+xmax)/2, (ymin+ymax)/2)
                if i in pedestrians:
                    pedestrians[i]["position"].append(center)
                    pedestrians[i]["frame"].append(eval(l[5]))
                    pedestrians[i]["lost"].append(eval(l[6]))
                    pedestrians[i]["occluded"].append(eval(l[7]))
                    pedestrians[i]["generated"].append(eval(l[8]))
                else:
                    update = {i: {"position": [center], "frame": [eval(l[5])],
                                  "lost": [eval(l[6])], "occluded": [eval(l[7])], "generated": [eval(l[8])]}}
                    pedestrians.update(update)
            id_prev = _id
        self.pedestrian_dict = pedestrians

    def __boundaries(self):
        self.boundary = [[(500, 2076), (561, 1162), (-30, 1129), (-42, 2045)],
                         [(734, 2082), (798, 1171), (1423, 1175), (1440, 2008)],
                         [(826, -59), (793, 892), (1446, 906), (1465, -57)],
                         [(636, -59), (600, 855), (-13, 827), (-27, -29)]]
        self.poly_bounds = []
        self.bounds = [[], [], [], []]
        for i, poly in enumerate(self.boundary):
            self.bounds[i] = list(zip(*poly[0:-1]))
            for p in poly[0:-1]:
                self.poly_bounds.append(Point(p))
        self.polygon = Polygon(self.poly_bounds)

    def plot_dataset(self, with_bounds=True):
        plt.figure()
        for i in self.pedestrian_dict.keys():
            centers = list(zip(*self.pedestrian_dict[i]["position"]))
            x, y = centers[0], centers[1]
            for (xp, yp) in zip(x, y):
                if Point(xp, yp).within(self.polygon):
                    color = "r"
                    break
                else:
                    color = "b"
            plt.scatter(x, y, c=color, s=0.1)
        if with_bounds:
            for plot in self.bounds:
                plt.plot(plot[0], plot[1], "g-", linewidth=3)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    data = SDD_Data()
    data.plot_dataset()
