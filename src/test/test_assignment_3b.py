import unittest
import numpy as np
from src.main.assignment_3b import (
    gaussian_elimination,
    backward_substitution,
    lu_factorization,
    is_diagonally_dominant,
    is_positive_definite
)

class TestAssignment3b(unittest.TestCase):
    def test_gaussian_elimination(self):
        A = np.array([[2, -1, 1, 6],
                      [1, 3, 1, 0],
                      [-1, 5, 4, -3]], dtype=float)
        gaussian_elimination(A)
        x = backward_substitution(A)
        expected = np.array([3, -1, 1])
        np.testing.assert_array_almost_equal(x, expected)

    def test_lu_factorization(self):
        A = np.array([[1, 1, 0, 3],
                      [2, 1, -1, 1],
                      [3, -1, -1, 2],
                      [-1, 2, 3, -1]], dtype=float)
        L, U = lu_factorization(A)
        self.assertAlmostEqual(np.prod(np.diag(U)), -13.0)

    def test_diagonally_dominant(self):
        A = np.array([[9, 0, 5, 2, 1],
                      [3, 9, 1, 2, 1],
                      [0, 1, 7, 2, 3],
                      [4, 2, 3, 12, 2],
                      [3, 2, 4, 0, 8]], dtype=float)
        self.assertTrue(is_diagonally_dominant(A))

    def test_positive_definite(self):
        A = np.array([[2, 2, 1],
                      [2, 3, 0],
                      [1, 0, 2]], dtype=float)
        self.assertTrue(is_positive_definite(A))

if __name__ == "__main__":
    unittest.main()
