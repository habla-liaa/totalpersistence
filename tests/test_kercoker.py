import numpy as np
from totalpersistence.totalpersistence import kercoker_via_cone, general_position_distance_matrix
from totalpersistence.utils import conematrix


def data_for_test():
    # Test input data

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
    Y = np.array([[0, 2], [1, 2], [2, 2]])
    f = np.array([0, 0, 1, 1, 2, 2])
    cone_eps = 1e-10
    tol = 1e-10

    return X, Y, f, cone_eps, tol


def test_cone_matrix():
    # Test input data
    X, Y, f, cone_eps, tol = data_for_test()

    np.random.seed(42)  # For reproducibility
    # Compute perturbed distance matrices
    dX = general_position_distance_matrix(X, perturb=1e-13)
    dY = general_position_distance_matrix(Y, perturb=1e-13)

    D = conematrix(dX, dY, f, cone_eps)
    # print("Cone Matrix:", D)

    expected_D = np.array([[0.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.41421356e+00, 2.00000000e+00, 
                            2.23606798e+00, 0.00000000e+00, 1.00000000e+00, 2.00000000e+00, 1.00000000e-10],
                           [1.00000000e+00, 0.00000000e+00, 1.41421356e+00, 1.00000000e+00, 2.23606798e+00,
                               2.00000000e+00, 0.00000000e+00, 1.00000000e+00, 2.00000000e+00, 1.00000000e-10],
                           [1.00000000e+00, 1.41421356e+00, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                               1.41421356e+00, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00, 1.00000000e-10],
                           [1.41421356e+00, 1.00000000e+00, 1.00000000e+00, 0.00000000e+00, 1.41421356e+00,
                            1.00000000e+00, 1.00000000e+00, 0.00000000e+00, 1.00000000e+00, 1.00000000e-10],
                           [2.00000000e+00, 2.23606798e+00, 1.00000000e+00, 1.41421356e+00, 0.00000000e+00,
                            1.00000000e+00, 2.00000000e+00, 1.00000000e+00, 0.00000000e+00, 1.00000000e-10],
                           [2.23606798e+00, 2.00000000e+00, 1.41421356e+00, 1.00000000e+00, 1.00000000e+00,
                            0.00000000e+00, 2.00000000e+00, 1.00000000e+00, 0.00000000e+00, 1.00000000e-10],
                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 1.00000000e+00, 2.00000000e+00,
                            2.00000000e+00, 0.00000000e+00, 5.00000000e-01, 1.00000000e+00, 3.23606798e+00],
                           [1.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
                            1.00000000e+00, 5.00000000e-01, 0.00000000e+00, 5.00000000e-01, 3.23606798e+00],
                           [2.00000000e+00, 2.00000000e+00, 1.00000000e+00, 1.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 1.00000000e+00, 5.00000000e-01, 0.00000000e+00, 3.23606798e+00],
                           [1.00000000e-10, 1.00000000e-10, 1.00000000e-10, 1.00000000e-10, 1.00000000e-10,
                            1.00000000e-10, 3.23606798e+00, 3.23606798e+00, 3.23606798e+00, 0.00000000e+00]])

    # Compare results
    assert D.shape == expected_D.shape, "Shape mismatch"
    assert np.allclose(
        D, expected_D, atol=tol), "Cone matrix does not match expected values"


def test_kercoker_via_cone():
    # Test input data
    X, Y, f, cone_eps, tol = data_for_test()

    np.random.seed(42)  # For reproducibility
    # Compute perturbed distance matrices
    dX = general_position_distance_matrix(X, perturb=1e-13)
    dY = general_position_distance_matrix(Y, perturb=1e-13)

    # Run the function
    coker_dgm, ker_dgm, cone_dgm, dgmX, dgmY = kercoker_via_cone(
        dX, dY, f, maxdim=2, cone_eps=cone_eps)

    expected_coker = np.array([[0.0, float("inf")]])
    expected_ker = [
        # np.array([[0.5, 1.0], [0.5, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        np.array([[0., 1.],  [0., 1.], [0., 1.]]),
        np.array([[1.0, 1.41421356], [1.0, 1.41421356]]),
    ]

    # Compare results
    assert len(coker_dgm) == 1
    assert np.allclose(coker_dgm[0], expected_coker, atol=tol)
    assert len(ker_dgm) == 2

    print("Expected Coker Diagram:", expected_coker)
    print("Coker Diagram:", coker_dgm)
    print("Expected Kernel Diagrams:", expected_ker)
    print("Kernel Diagrams:", ker_dgm)
    for r, e in zip(ker_dgm, expected_ker):
        assert np.allclose(r, e, atol=tol)
