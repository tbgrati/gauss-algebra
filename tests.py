import numpy as np
import unittest

from main import tridiagonal_optimized_gauss, basic_gauss, parcial_pivot_gauss

class TestGauss(unittest.TestCase):

    def test_tridiagonal_optimized_gauss(self):
        A = np.array([[3, 1, 0],
                      [3, 4, 1],
                      [0, 3, 3]], dtype=np.float64)
        b = np.array([5, 7, 6], dtype=np.float64)
        expected_x = np.array([5/3, 0, 2], dtype=np.float64)
        
        x = tridiagonal_optimized_gauss(A, b)
        
        np.testing.assert_allclose(x, expected_x, atol=1e-10, rtol=0)


    def test_parcial_pivot_gauss(self):
        A = np.array([[3, 1, 0],
                      [3, 4, 1],
                      [0, 3, 3]], dtype=np.float64)
        b = np.array([5, 7, 6], dtype=np.float64)
        expected_x = np.array([5/3, 0, 2], dtype=np.float64)
        
        x = parcial_pivot_gauss(A, b)
        
        np.testing.assert_allclose(x, expected_x, atol=1e-10, rtol=0)
        
    
    def test_basic_gauss(self):
        A = np.array([[3, 1, 0],
                      [3, 4, 1],
                      [0, 3, 3]], dtype=np.float64)
        b = np.array([5, 7, 6], dtype=np.float64)
        expected_x = np.array([5/3, 0, 2], dtype=np.float64)
        
        x = basic_gauss(A, b)
        
        np.testing.assert_allclose(x, expected_x, atol=1e-10, rtol=0)

if __name__ == '__main__':
    unittest.main()