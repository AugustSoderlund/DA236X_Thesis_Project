from zonotope import *
from operations import *
import numpy as np

if __name__ == "__main__":
    x=np.array([1,1])
    G=np.array([[1,0,1],[0,1,2]])
    p = np.array([0,-2])
    z1 = zonotope(x,G)
    boo = is_inside(z1, p)
    print(boo)
    visualize_zonotopes([z1])
    plt.show()