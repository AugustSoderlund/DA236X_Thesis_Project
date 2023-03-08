from zonotope import *
from operations import *
import numpy as np

if __name__ == "__main__":
    x=np.array([1,1])
    G=np.array([[1,0,0],[0,1,0]]).reshape(2,3)
    z1 = zonotope(x,G)
    z2 = zonotope(x+2, G)
    z3 = minkowski_sum(z1, z2)
    z4 = product(z1,z2)
    z3.color=(0.9, 0.9, 0.1)
    visualize_zonotopes([z1,z2,z3])