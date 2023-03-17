from DRA.zonotope import zonotope, matrix_zonotope
from DRA.operations import create_M_w, product, minkowski_sum, cartesian_product, reduce, visualize_zonotopes
from DRA.reachability import LTI_reachability
import matplotlib.pyplot as plt
from control.matlab import * 
import numpy as np
import random


"""
    This script is for testing the implementation such that it works.
    Compare the results from this implementation to the results as
    seen in 'Data-Driven Reachability Analysis Using Matrix Zonotopes'
"""

if __name__ == "__main__":
    # system dynamics
    dim_x = 5
    A = np.array([[-1,-4,0,0,0],[4,-1,0,0,0],[0,0,-3,1,0],[0,0,-1,-3,0],[0,0,0,0,1]])
    B_ss = np.ones(shape=(5,1))
    C = np.array([1,0,0,0,0])
    D = 0
    # define continuous time system
    sys_c = ss(A,B_ss,C,D)
    # convert to discrete system
    samplingtime = 0.05
    sys_d = c2d(sys_c,samplingtime)


    # Number of trajectories
    initpoints = 13
    # Number of time steps
    steps = 5
    initpoints = 1
    # Number of time steps
    steps = 120
    totalsamples = initpoints*steps
    # initial set and input
    X0 = zonotope(np.ones((dim_x,1)),0.1*np.eye(dim_x))
    U = zonotope(np.array([10]),np.array([0.25]))

    W = zonotope(np.zeros((dim_x,1)),0.005*np.ones((dim_x,1)))


    # NOTE: FIRST USAGE OF MY OWN FUNCTION
    Wmatzono = create_M_w(totalsamples, W)

    def randPoint(z):
        x, G = z.x, z.G
        lower_ = sum(np.diag(x) - G)
        upper_ = sum(np.diag(x) + G)
        return random.uniform(lower_, upper_)



    u = [0]*totalsamples
    for i in range(0,totalsamples):
        u[i] = randPoint(U)#random.uniform(9, 11)


    # simulate the system to get the data
    x0 = X0.x
    x = np.zeros(shape=(initpoints*dim_x, steps+1))
    x[0:x0.shape[0],0] = x0.reshape(x0.shape[0],)
    utraj = np.zeros(shape=(initpoints*dim_x, steps))
    index = 0
    for j in range(0, initpoints*dim_x, dim_x):
        
        x[j:j+dim_x,1] = randPoint(X0)
        for i in range(0,steps):
            utraj[j,i] = u[index]
            c = sys_d.A*x[j:j+dim_x,i].reshape(x0.shape[0],1) + sys_d.B*u[index] + randPoint(W)
            x[j:j+dim_x,i+1] = c.reshape(x0.shape[0],);      
            index += 1

    # concatenate the data trajectories 
    index_0 = 0
    index_1 = 0
    x_meas_vec_1 = np.zeros(shape=(x0.shape[0], initpoints*steps))
    x_meas_vec_0 = np.zeros(shape=(x0.shape[0], initpoints*steps))
    u_mean_vec_0 = np.zeros(shape=(x0.shape[0], initpoints*steps))
    for j in range(0,initpoints*dim_x,dim_x):
        for i in range(1,steps+1):
            x_meas_vec_1[:,index_1] = x[j:j+dim_x,i]
            index_1 += 1
        for i in range(0,steps):
            u_mean_vec_0[:,index_0] = utraj[j,i]
            x_meas_vec_0[:,index_0] = x[j:j+dim_x,i]
            index_0 += 1

    U_full = u_mean_vec_0[:,0:totalsamples]
    X_0T = x_meas_vec_0[:,0:totalsamples]
    X_1T = x_meas_vec_1[:,0:totalsamples]

    plt.plot(X_1T[0,:], X_1T[1,:])
    #plt.show()


    X1W_cen =  X_1T - Wmatzono.x
    X1W = matrix_zonotope(X1W_cen, Wmatzono.G)

    # set of A and B
    _stacked = np.vstack((X_0T, U_full))
    AB = product(X1W, np.linalg.pinv(_stacked))
    #print(X1W.G.shape, X1W.x.shape, _stacked.shape, np.linalg.pinv(_stacked).shape, AB.G.shape, AB.x.shape)

    # validate that A and B are within AB
    #intAB11 = intervalMatrix(AB);
    #intAB1 = intAB11.int;
    #intAB1.sup >= [sys_d.A,sys_d.B]
    #intAB1.inf <= [sys_d.A,sys_d.B]


    # set number of steps in analysis
    totalsteps = 5
    X_model = [0] * (totalsteps+1) # Maybe add 1 to both this and below
    X_data = [0] * (totalsteps+1)
    # init sets for loop
    X_model[0] = X0
    X_data[0] = X0

    for i in range(0,totalsteps):
        
        # 1) model-based computation
        X_model[i] = reduce(X_model[i],400)
        X_model[i+1] = minkowski_sum(minkowski_sum(product(sys_d.A, X_model[i]), product(sys_d.B, U)), W)
        # 2) Data Driven approach
        #X_data[i] = reduce(X_data[i],400)
        #c = cartesian_product(X_data[i], U)
        #print(c.G.shape, c.x.shape, AB.G.shape, AB.x.shape)
        #a = product(AB, cartesian_product(X_data[i], U))
        # TODO: There are some issues with the output from the cartesian product I think
        #   * Look in the CORA package for comparison
        #   * Debug this and ensure that it would still work for the main.py
        #c = cartesian_product(X_data[i], U)
        #print(X_data[i].G.shape, X_data[i].x.shape, c.G.shape, c.x.shape)
        #X_data[i+1] = minkowski_sum(product(AB, cartesian_product(X_data[i], U)), W)

    U = zonotope(10*np.ones(shape=(5,1)),0.25*np.ones(shape=(5,1)).reshape(5,1))
    X_data = LTI_reachability(U_full, X_1T, X_0T, X0, W, Wmatzono, U, N=totalsteps+1, n=400)
    for i in range(len(X_data)):
        X_data[i].color = [0.1,0.1,0.6]


    visualize_zonotopes([*X_data, *X_model], show=True)