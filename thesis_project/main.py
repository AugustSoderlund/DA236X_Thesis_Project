import time
from utils.visualization import *
from DRA.operations import visualize_zonotopes, optimize_vertices, create_M_w
from DRA.reachability import LTI_reachability
from PTC.input_state import *
from PTC.classification import DecisionTree
from utils.data_reader import SinD
from utils.data_processor import *
from utils.evaluation import *
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
    noise = 0.25
    #true_labels = _sind.labels(test_data)
    #print("Accuracy (Decision tree):    ", str(sum(true_labels==p_dt)/len(p_dt)))
    #classification_acc_per_class(true_labels, p_dt, plot=True)
    #visualize_all_classes(_sind.map, len(LABELS.keys()), train_data, p, 30)
    #visualize_class(_sind.map, 0, test_data, p_dt)
    d = separate_data_to_class(test_data, p_dt)
    c_z = np.array([2,2])
    G_z = np.array([[4,0,1],[0,4,2]])
    z = zonotope(c_z, G_z)
    v = np.array([-1,1])
    classification = p_dt[0]
    U, X_p, X_m = create_io_state(d, z, v, classification)
    U_all, X_p_all, X_m_all = create_io_state(d, z, v, [0,1,2,3,4])
    z_w = zonotope(np.array([0,0]), process_noise*np.ones(shape=(2,1)))
    M_w = create_M_w(U.shape[1], z_w)
    M_w_base = create_M_w(U_all.shape[1], z_w)
    U_k = []
    _trajs = U.reshape(-1,2,29)
    for i in range(0,29):
        _v = _trajs[:,:,i]
        _v = np.sum(_v, axis=0) / len(_v)
        _U = zonotope(c_z=_v, G_z=np.array([[noise,0],[0,noise]]))
        U_k.append(_U)
    U_k_all = []
    _trajs = U_all.reshape(-1,2,29)
    for i in range(0,29):
        _v = _trajs[:,:,i]
        _v = np.sum(_v, axis=0) / len(_v)
        _U = zonotope(c_z=_v, G_z=np.array([[noise,0],[0,noise]]))
        U_k_all.append(_U)
    #plt.scatter(X_p[0], X_p[1], c="r")
    z = zonotope(np.array([2,2]), np.array([[1,0,0.5],[0,1,0.5]]))
    U_k = zonotope(c_z=np.array([10,0]), G_z=np.array([[noise], [noise]]))
    U_k_all = zonotope(c_z=np.array([10,0]), G_z=np.array([[noise*2,0],[0,10*noise]]))
    print("Reachability for modal prediction")
    R = LTI_reachability(U, X_p, X_m, z, z_w, M_w, U_k, N=29)
    print("Baseline reachability")
    R_base = LTI_reachability(U_all, X_p_all, X_m_all, z, z_w, M_w_base, U_k_all, N=29)
    R = R[-1]
    R_base = R_base[-1]
    R.color = [0,0.6,0]
    R_base.color = [0.55,0.14,0.14]
    print("Reachability completed")
    print(zonotope_area(R))
    print(zonotope_area(R_base))
    z = zonotope(np.array([2,2]), np.array([[1,0,0.5],[0,1,0.5]]))
    visualize_zonotopes([z, R, R_base], map=_sind.map, show=True)
