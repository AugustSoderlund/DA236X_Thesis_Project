from dataclasses import dataclass
import numpy as np


@dataclass
class Zonotope:
    center: np.ndarray
    generator_matrix: np.ndarray

if __name__ == "__main__":
    z1 = Zonotope()