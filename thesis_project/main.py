from sklearn.model_selection import train_test_split
from utils.visualization import *
from PTC.clustering import AutoCluster, Cluster
from DRA.operations import visualize_zonotopes
from DRA.reachability import LTI_reachability
from PTC.input_state import *
from PTC.classification import DecisionTree
from utils.data_reader import SinD, LABELS
from DRA.zonotope import zonotope
import numpy as np
import pickle
import os


ROOT = os.getcwd()

def load_data(file: str = "sind.pkl") -> np.ndarray:
    _f = open(ROOT + "/thesis_project/.datasets/" + file, "rb")
    return pickle.load(_f)

def save_data(data: np.ndarray, file: str):
    _f = open(ROOT + "/thesis_project/.datasets/" + file, "rb")
    pickle.dump(data, _f)

def split_data(data: np.ndarray, test_size: float = 0.2):
    return train_test_split(data, test_size=test_size)

def label_data(sind: SinD, data: np.ndarray, input_len: int = 30, save: str = None) -> np.ndarray:
    if save: save_data(data, save)
    return sind.labels(data, input_len)


if __name__ == "__main__":
    _sind = SinD()
    if input("Load? (y/n)") == "y":
        data = load_data()
    else:
        data = _sind.data()
    train_data, test_data = split_data(data)
    labels = _sind.labels(train_data)
    dt = DecisionTree(max_depth=20)
    dt.train(train_data, labels)
    p_dt = dt.predict(test_data)
    #true_labels = _sind.labels(test_data)
    #print("Accuracy (Decision tree):    ", str(sum(true_labels==p_dt)/len(p_dt)))
    #classification_acc_per_class(true_labels, p_dt, plot=True)
    #visualize_all_classes(_sind.map, len(LABELS.keys()), train_data, p, 30)
    #visualize_class(_sind.map, 4, test_data, true_labels, 30)
    d = separate_data_to_class(test_data, p_dt)
    c_z = np.array([test_data[0][30],  test_data[0][60]])
    G_z = np.array([[4,0,3],[0,4,-2]])
    z = zonotope(c_z, G_z)
    classification = p_dt[0]
    U, X_p, X_m = create_io_state(d, z, classification)
    z_w = zonotope(np.array([0,0]), np.array([[0.01,0],[0,0.01]]))
    plt.scatter(X_p[0], X_p[1], c="r")
    visualize_zonotopes([z], map=_sind.map, show=True)
    #LTI_reachability(U, X_p, X_m, z, z_w, M_w, U_k)
