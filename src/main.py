import math
import numpy as np
import time

# Print float values with trailing 0s
np.set_printoptions(formatter={'float': '{:0.1f}'.format})


def create_tridiagonal_matrix(n):
    matrix = np.zeros((n, n))

    main_diag_values = np.round(np.random.uniform(0, 10, n), 1)  
    off_diag_values = np.round(np.random.uniform(0, 10, n - 1), 1)  

    np.fill_diagonal(matrix, main_diag_values)

    np.fill_diagonal(matrix[:-1, 1:], off_diag_values)

    np.fill_diagonal(matrix[1:, :-1], off_diag_values)

    return matrix


def create_matrix(n):
    matrix = np.round(np.random.uniform(0, 10, (n, n)), 1)
    return matrix


def measure_time(func, args):
    start_time = time.perf_counter()
    func(*args)
    end_time = time.perf_counter()
    return (end_time - start_time)


def basic_gauss(A,b):
    n = len(b)
    # Covert to augmented matrix with A | b solution
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Descending process
    for i in range(n):
        for j in range(i+1, n):
            if Ab[i, i] == 0:
                break
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # Ascending process
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

    return x


def parcial_pivot_gauss(A,b):
    n = len(b)
    # Augmented matrix
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination with partial pivoting
    for i in range(n):
        # Pivoting
        max_row = np.argmax(np.abs(Ab[i:n, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        for j in range(i+1, n):
            if Ab[i, i] == 0:
                break
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

    return x



def tridiagonal_optimized_gauss(A,b):
    n = len(b)
    
    ci = np.zeros(n-1)
    di = np.zeros(n)
    
    ci[0] = A[0, 1] / A[0, 0]
    di[0] = b[0] / A[0, 0]

    for i in range(1, n-1):
        ci[i] = A[i, i+1] / (A[i, i] - ci[i-1] * A[i, i-1])
    
    for i in range(1, n):
        di[i] = (b[i] - di[i-1] * A[i, i-1]) / (A[i, i] - ci[i-1] * A[i, i-1])

    x = np.zeros(n, dtype=np.float64)
    x[n-1] = di[n-1]

    for i in range(n-2, -1, -1):
        x[i] = di[i] - ci[i] * x[i+1]
    
    return x
