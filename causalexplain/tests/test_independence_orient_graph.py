import numpy as np
import pytest
import pandas as pd

from causalexplain.independence.edge_orientation import get_edge_orientation


def test_get_edge_orientation():
    # Create a mock DataFrame
    data = pd.DataFrame({
        'x': np.random.rand(100),
        'y': np.random.rand(100)
    })

    # Test with default parameters
    result = get_edge_orientation(data, 'x', 'y')
    assert isinstance(result, int)
    assert result in [-1, 0, 1]

    # Test with custom parameters
    result = get_edge_orientation(data, 'x', 'y', iters=50, method='gam', verbose=True)
    assert isinstance(result, int)
    assert result in [-1, 0, 1]

    # Test with invalid method
    with pytest.raises(ValueError):
        get_edge_orientation(data, 'x', 'y', method='invalid')

    # Test with non-existent columns
    with pytest.raises(KeyError):
        get_edge_orientation(data, 'a', 'b')
