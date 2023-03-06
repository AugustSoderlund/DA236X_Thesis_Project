from sklearn.model_selection import train_test_split
from utils.visualization import *
from PTC.clustering import AutoCluster, Cluster
from PTC.classification import DecisionTree
from utils.data_reader import SinD, LABELS
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
    _sind = SinD()
    if input("Load? (y/n)") == "y":
        data = load_data()
    else:
        data = _sind.data()
    train_data, test_data = split_data(data)
    labels = _sind.labels(train_data)
    c = DecisionTree()
    c.train(train_data, labels)
    p = c.predict(train_data)
    visualize_all_classes(_sind.map, len(LABELS.keys()), train_data, p, 30)