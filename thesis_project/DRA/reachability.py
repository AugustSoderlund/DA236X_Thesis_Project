import numpy as np
import pypolycontain as pp
from DRA.zonotope import zonotope
from typing import List
if __package__ or "." in __name__:
    from .operations import *
else:
    from operations import *


def LTI_reachability(U_minus: np.ndarray, X_plus: np.ndarray, X_minus: np.ndarray, 
                    X_0: pp.zonotope, Z_w: pp.zonotope, M_w: pp.zonotope, 
                    U_k: Union[pp.zonotope, List[pp.zonotope]], N: int = 30, n: int = 50) -> List[pp.zonotope]:
    """ Linear time-invariant reachability analysis

        Parameters:
        -----------
        U_minus : np.ndarray
            Input array of all inputs in each trajectory, looking like
            [u(1)(0) . . . u(1)(T1-1) . . . u(K)(0) . . . u(K)(TK-1)]
        X_plus : np.ndarray
            State array of all states in each trajectory from T=1 
            (instead of T=0 as start), on the form
            [x(1)(1) . . . x(1)(T1) . . . x(K)(1) . . . x(K)(TK )]
        X_minus : np.ndarray
            State array of all states in each trajectory to T-1, on the form
            [x(1)(0) . . . x(1)(T1-1) . . . x(K)(0) . . . x(K)(TK-1)]
        X_0 : pp.zonotope
            Initial state represented as a zonotope
        Z_w : pp.zonotope
            Process noise zonotope
        M_w : pp.zonotope
            Concatenation of multiple noise zonotopes
        U_k : pp.zonotope
            Input zonotope
        N : int (default = 30)
            Number of timesteps
        n : int (default = 100)
            Reduced order of reachable sets
    """
    if type(U_k) == pp.zonotope: U_k = [U_k] * N
    R = [0] * N
    R[0] = reduce(X_0, order=n)
    _stacked = np.vstack([X_minus, U_minus])
    _X = matrix_zonotope(X_plus - M_w.x, M_w.G)
    M_sigma = product(_X, np.linalg.pinv(_stacked))
    for i in range(0, N-1):
        R[i+1] = minkowski_sum(product(M_sigma, cartesian_product(R[i], U_k[i])), Z_w)
        R[i+1] = reduce(R[i+1], order=n)
    return R

def bootstrap_RA(_args: dict, conf: np.ndarray, conf_thresh: float = 0.1, N: int = None):
    """ Linear time-invariant reachability analysis

        Parameters:
        -----------
        _args : dict
            All arguments required for the LTI_reachability 
            function for each class
        conf : np.ndarray
            Confidence in each class from the classification
        conf_thresh : float (default = 0.1)
            The threshold for the confidence for the classes 
            that the reachability will consider
        N : int
            The time horizon (in time steps, should be set in 
            accordance with the frequency of measurements)
    """
    _R = [0] * len(conf)
    conf[conf < conf_thresh] = 0
    _N = [round(N*_c) for _c in conf/sum(conf)]
    for i,n in enumerate(_N):
        _R[i] = LTI_reachability(*_args[i], n)[-1]
    _z = pp.zonotope(G=np.ndarray([[0,0],[0,0]]), x=np.ndarray([0,0]))
    for _r in _R:
        _z = minkowski_sum(_z, _r)
    return _z



def final_constraint(map, R_k: pp.zonotope):
    pass