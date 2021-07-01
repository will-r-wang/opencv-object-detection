import numpy as np

def count_subset_occurrences(array, subset_array):
    occurrences = 0
    for idx in range(len(array) - len(subset_array) + 1):
        if np.array_equal(array[idx:(idx + len(subset_array))], subset_array):
            occurrences += 1
    return occurrences

def test_base_case():
    assert count_subset_occurrences(
        np.array([0, 1, 1, 1, 2, 2, 2, 1, 1, 3, 3, 3]), 
        np.array([1, 1])
    ) == 3

test_base_case()
