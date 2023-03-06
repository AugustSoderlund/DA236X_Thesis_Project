from PTC.clustering import AutoCluster
from utils.data_reader import SinD
import pickle
import os


ROOT = os.getcwd()

def load_data(file: str = "sind.pkl"):
    _f = open(ROOT + "/thesis_project/.datasets/" + file, "rb")
    return pickle.load(_f)

if __name__ == "__main__":
    if input("Load? (y/n)") == "y":
        data = load_data()
    else:
        _sind = SinD()
        data = _sind.data()
    c = AutoCluster()
    c.train(data)