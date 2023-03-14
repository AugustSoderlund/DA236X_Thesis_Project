from zonotope import *
from operations import *
import numpy as np

if __name__ == "__main__":
    x1=np.array([5,3]).reshape(2,1)
    x2=np.array([0,0.5]).reshape(2,1)
    x3=np.array([0,-0.5]).reshape(2,1)
    #x=np.concatenate((x1,x2),axis=1)
    G1=np.array([[1,0,0.5],[0,1,0.5]])
    G2=np.array([[0,0],[0.5,0]])
    #G = np.concatenate((G1,G2),axis=1)
    p = np.array([0,-2])
    z1 = zonotope(x1,G1)
    z2 = zonotope(x2,G2)
    z3 = minkowski_sum(z1,z2)
    z4 = product(z1,z2)
    print(z4.x, z4.G)
    visualize_zonotopes([z1], show=True, scale_axes=True)
    