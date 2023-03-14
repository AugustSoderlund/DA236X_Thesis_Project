import time
from utils.visualization import *
from DRA.operations import visualize_zonotopes, optimize_vertices
from DRA.reachability import LTI_reachability
from PTC.input_state import *
from PTC.classification import DecisionTree
from utils.data_reader import SinD
from utils.data_processor import *
from DRA.zonotope import zonotope
import numpy as np

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
    process_noise = 0.01
    noise = 0.01
    #true_labels = _sind.labels(test_data)
    #print("Accuracy (Decision tree):    ", str(sum(true_labels==p_dt)/len(p_dt)))
    #classification_acc_per_class(true_labels, p_dt, plot=True)
    #visualize_all_classes(_sind.map, len(LABELS.keys()), train_data, p, 30)
    #visualize_class(_sind.map, 0, test_data, p_dt)
    d = separate_data_to_class(test_data, p_dt)
    c_z = np.array([test_data[0][30],  test_data[0][60]])
    G_z = np.array([[4,0,3],[0,4,-2]])
    z = zonotope(c_z, G_z)
    classification = p_dt[0]
    U, X_p, X_m = create_io_state(d, z, classification)
    z_w = zonotope(np.array([0,0]), np.array([[process_noise,0],[0,process_noise]]))
    C_M = np.array([])
    for i in range(X_p.shape[1]):
        _C_M = np.array([noise, noise]).reshape(2,1)
        C_M = np.hstack([C_M, _C_M]) if C_M.size else _C_M
    M_w = zonotope(c_z=C_M, G_z=np.array([[0,0],[0,0]]))
    U_k = zonotope(c_z=np.array([0,0]), G_z=np.array([[1,0],[0,1]]))
    #plt.scatter(X_p[0], X_p[1], c="r")
    R = LTI_reachability(U, X_p, X_m, z, z_w, M_w, U_k, N=30)
    R = R[-1]
    #V_opt = optimize_vertices(R)
    R.color = [0.9,0.9,0.1]
    visualize_zonotopes([R, z], map=_sind.map, show=True)
