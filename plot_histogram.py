import numpy as np

def plot_histogram(array):
    histogram = {}
    for idx in array:
        histogram[idx] = histogram.get(idx, 0) + 1
    return histogram

def test_base_case(): 
    assert plot_histogram(np.array([0, 1, 1, 1, 2, 2, 2, 1, 1, 3, 3, 3])) == {0: 1, 1: 5, 2: 3, 3: 3}

test_base_case()
