from zonotope import *
from operations import *
import numpy as np

if __name__ == "__main__":
    x1=np.array([0,0]).reshape(2,1)
    x2=np.array([0,0]).reshape(2,1)
    #x=np.concatenate((x1,x2),axis=1)
    G1=np.array([[1,0],[0,1]])
    G2=np.array([[0,0],[0,0]])
    #G = np.concatenate((G1,G2),axis=1)
    p = np.array([0,-2])
    z1 = zonotope(x1,G1)
    z2 = zonotope(x2,G2)
    z3 = minkowski_sum(z1,z2)
    z2.color = [0.1,0.1,0.1]
    z3.color=[0.9,0.9,0.2]
    