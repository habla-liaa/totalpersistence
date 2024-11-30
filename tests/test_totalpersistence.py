import numpy as np
from totalpersistence import totalpersistence


def test_totalpersistence_basic():
    # Create simple test data
    X = np.array([[0, 0], [1, 0], [0, 1]])  # Triangle vertices
    Y = np.array([[0, 0], [2, 0], [0, 2]])  # Scaled triangle vertices
    f = np.array([0, 1, 2])  # Simple function values

    # Call totalpersistence
    coker_bottleneck_distances, ker_bottleneck_distances, coker_matchings, ker_matchings = totalpersistence(X, Y, f)

    # TODO: Add assertions for a relevant case

    # Basic assertions
    assert isinstance(coker_bottleneck_distances, list)
    assert isinstance(ker_bottleneck_distances, list)
    assert isinstance(coker_matchings, list)
    assert isinstance(ker_matchings, list)

    # The distance should be non-negative
    assert all(d >= 0 for d in coker_bottleneck_distances)
    assert all(d >= 0 for d in ker_bottleneck_distances)
