from sklearn.model_selection import train_test_split
from utils.visualization import *
from PTC.clustering import AutoCluster, Cluster
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
    c = Cluster(n=len(LABELS.keys()))
    c.train(train_data, labels)
    p = c.predict(test_data)
    print(type(c.classifier.labels_), type(labels))
    visualize_class(_sind.map, 0, train_data, labels, 30)