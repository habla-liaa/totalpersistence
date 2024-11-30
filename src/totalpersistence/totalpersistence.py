import numpy as np
from ripser import ripser
from scipy.spatial.distance import squareform
from .utils import (
    lipschitz,
    general_position_distance_matrix,
    to_condensed_form,
    conematrix,
    kercoker_bars,
    matrix_size_from_condensed,
)


def totalpersistence(X, Y, f, maxdim=1, eps=0, tol=1e-11, perturb=1e-7):
    dX = general_position_distance_matrix(X, perturb)
    dY = general_position_distance_matrix(Y, perturb)

    data, dgm, dgmX, dgmY = kercoker_via_cone(dX, dY, f, maxdim, eps, tol, perturb)

    distance_bottleneck, matching = persim.bottleneck(dgm_clean, dgm_noisy, matching=True)

    return data, dgm, dgmX, dgmY


def kercoker_via_cone(dX, dY, f, maxdim=1, cone_eps=0, tol=1e-11):
    """
    TODO: Compute the total persistence diagram using the cone algorithm.

    Parameters
    ----------
    dX : np.array
        Distance matrix of the source space in condensed form.
    dY : np.array
        Distance matrix of the target space in condensed form.
    f : np.array
        Function values.
    """

    n = matrix_size_from_condensed(dX)
    m = matrix_size_from_condensed(dY)

    f = np.array(f)

    # dY_ff = d(f(x_i),f(x_j)) para todo i,j
    i, j = np.triu_indices(n, k=1)
    f_i, f_j = f[i], f[j]
    f_pos = to_condensed_form(f_i, f_j, m)
    dY_ff = dY[f_pos.astype(int)]

    # dY_fy = d(f(x_i),y_j) para todo i,j
    indices = np.indices((n, m))
    i = indices[0].flatten()
    j = indices[1].flatten()
    f_i = f[i]
    DY_fy = np.zeros((n, m))
    DY_fy[i, j] = squareform(dY)[f_i, j]

    L = lipschitz(dX, dY_ff)
    dY = dY / L

    D = conematrix(squareform(dX), squareform(dY), DY_fy, cone_eps)

    dgmX = ripser(squareform(dX), distance_matrix=True, maxdim=maxdim)["dgms"]
    dgmY = ripser(squareform(dY), distance_matrix=True, maxdim=maxdim)["dgms"]
    cone_dgm = ripser(D, maxdim=maxdim, distance_matrix=True)["dgms"]

    coker_dgm, ker_dgm = kercoker_bars(cone_dgm, dgmX, dgmY, cone_eps, tol)
    return coker_dgm, ker_dgm, cone_dgm, dgmX, dgmY
