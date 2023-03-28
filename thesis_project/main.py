import time
from utils.visualization import *
from DRA.operations import visualize_zonotopes, optimize_vertices, create_M_w, input_zonotope
from DRA.reachability import LTI_reachability
from PTC.input_state import *
from PTC.classification import DecisionTree
from PTC.dnn_classifier import DNN, one_hot_encode
from utils.data_reader import SinD, LABELS
from utils.data_processor import *
from utils.evaluation import *
from DRA.zonotope import zonotope
import numpy as np
from descartes import PolygonPatch
from shapely.ops import linemerge, unary_union, polygonize


def func():
    _sind = SinD()
    if input("Load? (y/n)") == "y":
        data = load_data()
    else:
        data = _sind.data(input_len=60)
    train_data, test_data = split_data(data)
    labels = _sind.labels(train_data, input_len=60)
    train_data, labels = structure_input_data(train_data, labels)
    train_data, labels = structure_input_data(train_data, labels)
    dt = DecisionTree(max_depth=20)
    dt.train(train_data, labels, input_len=60)
    p_dt = dt.predict(test_data, input_len=60)
    print("Trained classifier")
    process_noise = 0.01
    noise = 0.25
    #true_labels = _sind.labels(test_data)
    #print("Accuracy (Decision tree):    ", str(sum(true_labels==p_dt)/len(p_dt)))
    #classification_acc_per_class(true_labels, p_dt, plot=True)
    #visualize_all_classes(_sind.map, len(LABELS.keys()), train_data, p, 30)
    #visualize_class(_sind.map, 0, test_data, p_dt)
    d = separate_data_to_class(train_data, labels)
    c_z = np.array([2,2])
    G_z = np.array([[4,0,1],[0,4,2]])
    z = zonotope(c_z, G_z)
    v = np.array([-1,1])
    classification = p_dt[0]
    U, X_p, X_m = create_io_state(d, z, v, classification, input_len=60)
    U_all, X_p_all, X_m_all = create_io_state(d, z, v, [0,1,2,3,4], input_len=60)
    print("IO states created")
    z_w = zonotope(np.array([0,0]), process_noise*np.ones(shape=(2,1)))
    M_w = create_M_w(U.shape[1], z_w)
    M_w_base = create_M_w(U_all.shape[1], z_w)
    U_k = []
    _trajs = U.reshape(-1,2,59)
    print("Noise zonotopes created")
    for i in range(0,29):
        _v = _trajs[:,:,i]
        _v = np.sum(_v, axis=0) / len(_v)
        _U = zonotope(c_z=_v, G_z=np.array([[noise,0],[0,noise]]))
        U_k.append(_U)
    U_k_all = []
    _trajs = U_all.reshape(-1,2,59)
    for i in range(0,59):
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



def _cut_poly_by_line(polygon: Polygon, line: LineString):
    merged = linemerge([polygon.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)


if __name__ == "__main__":
    input_len = 90
    a = input_len-1
    N = 30
    #func()
    _sind = SinD()
    if True:#input("Load? (y/n)") == "y":
        data = load_data()
        labels = load_data("sind_labels.pkl")
        train_data, test_data, train_labels, test_labels = split_data(data, labels)
    else:
        data = _sind.data(input_len=input_len)
        labels = _sind.labels(data, input_len=input_len)
        train_data, test_data, train_labels, test_labels = split_data(data, labels)
    train_data, train_labels = structure_input_data(train_data, train_labels)
    inp_size, out_size = train_data.shape[1], len(LABELS.keys())
    dnn = DNN(input_size=inp_size, output_size=out_size, nodes=[300, 150])
    dnn.train(train_data, train_labels, epochs=300)
    test_labels_one_hot = one_hot_encode(test_labels)
    test_labels_pred = dnn.predict(test_data)
    _labels = []
    for l in test_labels_pred:
        _labels.append(np.argmax(l))
    _labels = np.array(_labels)
    classification_acc_per_class(test_labels, _labels)
    dnn.plot_training()
    time.sleep(100)
    d = separate_data_to_class(train_data, train_labels)
    c_z = np.array([-3.4, 28.3])
    G_z = np.array([[4,0,1],[0,2,1]])
    #G_z = np.array([[0.5,0,0.25],[0,0.5,0.15]])
    z = zonotope(c_z, G_z)
    z_big = zonotope(c_z, G_z)
    v = np.array([1,0])
    classification = 1
    U, X_p, X_m, U_i = create_io_state(d, z, v, classification, input_len=input_len, drop_equal=True)
    U_all, X_p_all, X_m_all, U_i_all = create_io_state(d, z, v, [0,1,2,3,4,5,6], input_len=input_len, drop_equal=True)
    #U_all, X_p_all, X_m_all = U_all[:,U_all.shape[1]-U.shape[1]:], X_p_all[:,X_p_all.shape[1]-X_p.shape[1]:], X_m_all[:,X_m_all.shape[1]-X_m.shape[1]:]
    #U, X_p, X_m = U[:,0:a], X_p[:,0:a], X_m[:,0:a]
    #U_all, X_p_all, X_m_all = U_all[:,b*a:c*a], X_p_all[:,b*a:c*a], X_m_all[:,b*a:c*a]
    #U_all, X_p_all, X_m_all = U_all[:,np.random.randint(0,U_all.shape[1],size=U.shape[1])], X_p_all[:,np.random.randint(0,U_all.shape[1],size=U.shape[1])], X_m_all[:,np.random.randint(0,U_all.shape[1],size=U.shape[1])]
    D = d[classification]
    # plt.scatter(z.x[0], z.x[1], c="r")
    # for traj in D:
    #     _x, _y = traj[0:input_len], traj[input_len:2*input_len]
    #     plt.plot(_x,_y, c="g")
    # plt.show()
    process_noise = 0.005
    noise = 0.005
    # _,ax1 = plt.subplots()
    # ax1.scatter(z.x[0], z.x[1], c="r")
    # ax1.plot(X_p[0,:], X_p[1,:], c="g")
    # plt.show()
    v_x = []
    v_y = []
    X_p_traj, X_m_traj, U_traj = split_io_to_trajs(X_p, X_m, U, threshold=5, dropped=True, N=a)
    X_p_all_traj, X_m_all_traj, U_all_traj = split_io_to_trajs(X_p_all, X_m_all, U_all, threshold=5, dropped=True, N=a)
    U_k = input_zonotope(U_traj, N=a)
    U_k_all = input_zonotope(U_all_traj, N=a)
    print(len(U_traj), len(U_all_traj))
    #_trajs = U.reshape(-1,2,a)
    #vx, vy = U[0,:].reshape(-1,a), U[1,:].reshape(-1,a)
    # for i in range(0,a):
    #     _vel_x = np.mean(vx[:,i], axis=0)
    #     _vel_y = np.mean(vy[:,i], axis=0)
    #     #_vel = _vel / np.linalg.norm(_vel) * np.linalg.norm(_v,axis=0) / _v.shape[1]
    #     #_U = zonotope(c_z=_vel, G_z=0*np.eye(2))
    #     v_x.append(_vel_x)
    #     v_y.append(_vel_y)
    _,ax = plt.subplots()
    t = list(range(0,a))
    # for i in range(a, U.shape[1], a):
    #     _vx, _vy = U[0,i-a:i], U[1,i-a:i]
    #     ax.plot(t, _vx, c="r")
    #     ax.plot(t, _vy, c="g")
    for i,u_k in enumerate(U_k):
        ax.scatter(i,u_k.x[0], c="orange", s=10)
        ax.scatter(i,u_k.x[1], c="orange", s=10)
    #ax.plot(t,v_x, c="b", linewidth=2)
    #ax.plot(t,v_y, c="b", linewidth=2)
    _,ax2 = plt.subplots()
    for x in X_p_traj:
        ax2.plot(x[0,:], x[1,:])
    plt.show()
    z_w = zonotope(np.array([0,0]), process_noise*np.ones(shape=(2,1)))
    M_w = create_M_w(U.shape[1], z_w)
    M_w_base = create_M_w(U_all.shape[1], z_w)
    #U_k = []
    #vx, vy = U[0,:].reshape(-1,a), U[1,:].reshape(-1,a)
    #for i in range(0,a):
    #    _vel_x = np.mean(vx[:,i], axis=0)
    #    _vel_y = np.mean(vy[:,i], axis=0)
    #    _std_x = np.std(vx[:,i], axis=0)
    #    _std_y = np.std(vy[:,i], axis=0)
        #_vel = _vel / np.linalg.norm(_vel) * np.linalg.norm(_v,axis=0) / _v.shape[1]
    #    _U = zonotope(c_z=np.array([_vel_x, _vel_y]), G_z=np.array([[_std_x,0],[0,_std_y]]))
    #    U_k.append(_U)
    #U_k_all = []
    #vx, vy = U_all[0,:].reshape(-1,a), U_all[1,:].reshape(-1,a)
    #for i in range(0,a):
    #    _vel_x = np.mean(vx[:,i], axis=0)
    #    _vel_y = np.mean(vy[:,i], axis=0)
    #    _std_x = np.std(vx[:,i], axis=0)
    #    _std_y = np.std(vy[:,i], axis=0)
    #    #_vel = _vel / np.linalg.norm(_vel) * np.linalg.norm(_v,axis=0) / _v.shape[1]
    #    _U = zonotope(c_z=np.array([_vel_x, _vel_y]), G_z=np.array([[_std_x,0],[0,_std_y]]))
    #    U_k_all.append(_U)
    #plt.scatter(X_p[0], X_p[1], c="r")
    #print(U.shape, U_all.shape)
    #c_z = np.array([27.4,3.6])
    G_z = np.array([[0.5,0,0.25],[0,0.5,0.15]])
    z = zonotope(c_z, G_z)
    #v = np.array([sum(U[0,:])/U.shape[1], sum(U[1,:])/U.shape[1]])
    #v = v / np.linalg.norm(v) * 1.42
    #v_all = np.array([sum(U_all[0,:])/U_all.shape[1], sum(U_all[1,:])/U_all.shape[1]])
    #v_all = v_all / np.linalg.norm(v_all) * 1.42
    #U_k = zonotope(c_z=v, G_z=np.array([[noise], [noise]]))
    #U_k_all = zonotope(c_z=v_all, G_z=np.array([[noise],[noise]]))
    t = time.time()
    R = LTI_reachability(U, X_p, X_m, z, z_w, M_w, U_k, N=a)
    print("Reachability execution time: ", time.time()-t)
    #print("Baseline reachability")
    R_base = LTI_reachability(U_all, X_p_all, X_m_all, z, z_w, M_w_base, U_k_all, N=a)
    R = R[-1]
    R_base = R_base[-1]
    R.color = [0,0.6,0]
    R_base.color = [0.55,0.14,0.14]
    print("Reachability completed")
    print("Area of zonotope: ", round(zonotope_area(R), 4), " m^2")
    print("Area of (baseline) zonotope: ", round(zonotope_area(R_base), 4), " m^2")
    z = zonotope(c_z, G_z)
    visualize_zonotopes([z, R, R_base], map=_sind.map, show=False)
    x, y = [],[]
    _ped_poly = Polygon(pp.to_V(z))
    # for i,p in enumerate(X_p_all.T):
    #     _x, _y = p[0], p[1]
    #     x.append(_x)
    #     y.append(_y)
    #     if i != 0 and i % (a-1) == 0:
    #         if Point((_x,_y)).within(_ped_poly):
    #             plt.plot(x,y, c="r")
    #         x, y = [], []
    for i in range(a,X_p_all.shape[1],a):
        _x, _y = X_p_all[0,i-a:i], X_p_all[1,i-a:i]
        _x, _y = _x[-1], _y[-1]
        plt.scatter(_x,_y, c="r", s=2)
    for i in range(a,X_p.shape[1],a):
        _x, _y = X_p[0,i-a:i], X_p[1,i-a:i]
        _x, _y = _x[-1], _y[-1]
        plt.scatter(_x,_y, c="g", s=2)
    #plt.arrow(c_z[0], c_z[1], v[0], v[1])
    #plt.plot(X_p_all[0,:], X_p_all[1,:], c="r")
    #plt.plot(X_p[0,:], X_p[1,:], c="b")
    plt.show()
    #visualize_zonotopes([z], map=_sind.map, show=True)
    #train_data, labels = structure_input_data(train_data, labels)
    #dt = DecisionTree(max_depth=20)
    #dt.train(train_data, labels, input_len=60)
    #p_dt = dt.predict(test_data, input_len=60)
    #print("Trained classifier")
    #process_noise = 0.01
    #noise = 0.25
    #true_labels = _sind.labels(test_data)
    #print("Accuracy (Decision tree):    ", str(sum(true_labels==p_dt)/len(p_dt)))
    #classification_acc_per_class(true_labels, p_dt, plot=True)
    #visualize_all_classes(_sind.map, len(LABELS.keys()), train_data, p, 30)
    #visualize_class(_sind.map, 0, test_data, p_dt)
    #d = separate_data_to_class(train_data, labels)
    #print(d, len(d))
    #print(l, len(l))
    #_sind.plot_dataset()
    # visualize_class(_sind.map, LABELS["cross_right"], train_data, train_labels, input_len=input_len)
    # visualize_class(_sind.map, LABELS["cross_left"], train_data, labels, input_len=input_len)
    # visualize_class(_sind.map, LABELS["cross_straight"], train_data, labels, input_len=input_len)
    # visualize_class(_sind.map, LABELS["not_cross"], train_data, labels, input_len=input_len)
    # visualize_class(_sind.map, LABELS["cross_illegal"], train_data, labels, input_len=input_len)
    # visualize_class(_sind.map, LABELS["crossing_now"], train_data, labels, input_len=input_len)
    # visualize_class(_sind.map, LABELS["unknown"], train_data, labels, input_len=input_len)
    #plt.show()
    # dt = DecisionTree(max_depth=80)
    # dt.train(train_data, labels, input_len=input_len)
    # _l = dt.predict(test_data, input_len=input_len)
    # test_labels = _sind.labels(test_data, input_len=input_len)
    #visualize_class(_sind.map, LABELS["cross_right"], train_data, labels, input_len=60)
    #visualize_class(_sind.map, LABELS["cross_left"], train_data, labels, input_len=60)
    #visualize_class(_sind.map, LABELS["cross_straight"], train_data, labels, input_len=60)
    #visualize_class(_sind.map, LABELS["not_cross"], train_data, labels, input_len=60)
    #visualize_class(_sind.map, LABELS["cross_illegal"], train_data, labels, input_len=60)
    #visualize_class(_sind.map, LABELS["crossing_now"], train_data, labels, input_len=60)
    #visualize_class(_sind.map, LABELS["unknown"], train_data, labels, input_len=60)
    #classification_acc_per_class(test_labels, _l)
    """ p1 = Polygon(_sind.map.crosswalk_poly.boundary[0])
    p2 = Polygon(_sind.map.crosswalk_poly.boundary[1])
    _,ax = plt.subplots()
    _points = _sind.map.get_area("")
    l1 = LineString([(0,1),(27,31)])
    l2 = LineString([(27,3),(0,31)])
    p3s = _cut_poly_by_line(p1, l1)
    p4 = []
    for p3 in p3s:
        new_p = _cut_poly_by_line(p3,l2)
        for p in new_p:
            p4.append(p.difference(p2))
        #p4 = [*p4, *_cut_poly_by_line(p3,l2)]
        #p4.append(*_cut_poly_by_line(p3,l2))
    ax.scatter(*zip(*_points), alpha=0) # To get bounds correct
    for p in p4:
        ax.add_patch(PolygonPatch(p, alpha=0.2, color="r"))
    #ax.add_patch(PolygonPatch(p2, alpha=0.2, color="g"))
    plt.show()
    #plt.show() """