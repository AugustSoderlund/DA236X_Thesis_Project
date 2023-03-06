from sklearn.model_selection import train_test_split
from PTC.clustering import AutoCluster
from utils.data_reader import SinD
import numpy as np
import pickle
import os


ROOT = os.getcwd()

def load_data(file: str = "sind.pkl"):
    _f = open(ROOT + "/thesis_project/.datasets/" + file, "rb")
    return pickle.load(_f)

def split_data(data: np.ndarray, test_size: float = 0.2):
    return train_test_split(data, test_size=test_size)


if __name__ == "__main__":
    if input("Load? (y/n)") == "y":
        data = load_data()
    else:
        _sind = SinD()
        data = _sind.data()
    train_data, test_data = split_data(data)
    c = AutoCluster()
    c.train(train_data)
    p = c.predict(test_data)