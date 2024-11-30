import numpy as np
from totalpersistence.totalpersistence import kercoker_via_cone, general_position_distance_matrix


def test_kercoker_via_cone():
    # Test input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
    Y = np.array([[0, 2], [1, 2], [2, 2]])
    f = np.array([0, 0, 1, 1, 2, 2])

    # Compute perturbed distance matrices
    dX = general_position_distance_matrix(X, perturb=1e-13)
    dY = general_position_distance_matrix(Y, perturb=1e-13)

    # Run the function
    result = kercoker_via_cone(dX, dY, f, maxdim=2, cone_eps=0)[0]
    print(result)

    # Expected output
    expected = [
        {"dim": 0, "set": "coker", "b": 0.0, "d": float("inf")},
        {"dim": 0, "set": "ker", "b": 0.5, "d": 1.0},
        {"dim": 0, "set": "ker", "b": 0.5, "d": 1.0},
        {"dim": 0, "set": "ker", "b": 0.0, "d": 1.0},
        {"dim": 0, "set": "ker", "b": 0.0, "d": 1.0},
        {"dim": 0, "set": "ker", "b": 0.0, "d": 1.0},
        {"dim": 1, "set": "ker", "b": 1.0, "d": 1.4142135381698608},
        {"dim": 1, "set": "ker", "b": 1.0, "d": 1.4142135381698608},
    ]
    tol = 1e-10

    # Compare results
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert r["dim"] == e["dim"]
        assert r["set"] == e["set"]
        assert abs(r["b"] - e["b"]) < tol
        assert abs(r["d"] - e["d"]) < tol if e["d"] != float("inf") else r["d"] == float("inf")
