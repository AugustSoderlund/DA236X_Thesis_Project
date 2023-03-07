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
    dt = DecisionTree(max_depth=40)
    km = Cluster()
    km2 = AutoCluster(n=4)
    dt.train(train_data, labels)
    km.train(train_data, labels)
    km2.train(train_data)
    p_dt = dt.predict(test_data)
    p_km = km.predict(test_data)
    p_km2 = km2.predict(test_data)
    true_labels = _sind.labels(test_data)
    print("Accuracy (Decision tree):    ", str(sum(true_labels==p_dt)/len(p_dt)))
    print("Accuracy (Clustering):       ", str(sum(true_labels==p_km)/len(p_km)))
    print("Accuracy (AutoClustering):   ", str(sum(true_labels==p_km2)/len(p_km2)))
    classification_acc_per_class(true_labels, p_dt, plot=True)
    #visualize_all_classes(_sind.map, len(LABELS.keys()), train_data, p, 30)
    #visualize_class(_sind.map, 0, test_data, p, 30)