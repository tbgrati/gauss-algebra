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


def create_array(n):
    random_array = np.random.uniform(0.1, 9.9, size=n)
    return random_array


def create_matrix(n):
    matrix = np.round(np.random.uniform(0, 10, (n, n)), 1)
    return matrix

def multiply_matrix_by_vector(A, x):
    result_dot = np.dot(A, x)
    return result_dot


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

    # Descending process with partial pivoting
    for i in range(n):
        # Pivoting
        max_row = np.argmax(np.abs(Ab[i:n, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
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



def tri_time_dif(n):
    tri_n_matrix = create_tridiagonal_matrix(n)
    tri_n_array = create_array(n)
    basic_gauss_time = measure_time(basic_gauss, (tri_n_matrix, tri_n_array))
    optimized_gauss_time = measure_time(tridiagonal_optimized_gauss, (tri_n_matrix, tri_n_array))

    return basic_gauss_time - optimized_gauss_time


# prints cumulative precision diff between basic gauss and parcial pivot gauss
def precision_dif(n):
    n_matrix = create_matrix(n)
    n_array = create_array(n)

    basic_gauss_x = basic_gauss(n_matrix, n_array)
    parcial_pivot_x = parcial_pivot_gauss(n_matrix, n_array)

    basic_est = multiply_matrix_by_vector(n_matrix, basic_gauss_x)
    pivot_est = multiply_matrix_by_vector(n_matrix, parcial_pivot_x)

    basic_cumulative_error = 0
    pivot_cumulative_errir = 0

    for i in range(1,n):
        basic_cumulative_error += abs(n_array[i] - basic_est[i])
        pivot_cumulative_errir += abs(n_array[i] - pivot_est[i])

    print(basic_cumulative_error)
    print(pivot_cumulative_errir)
        


def main():

    print("Time comparison between basic Gauss and optimized Gauss for tridiagonal matrix")

    print("\nTime difference for 100x100 tridiagonal matrix:")
    print(tri_time_dif(100))
    print("Time difference for 1000x1000 tridiagonal matrix:")
    print(tri_time_dif(1000))

    print("\nPrecision comparison between 30x30 and 50x50 matrix")

    print("\nPrecision diff for 30x30 matrix")
    precision_dif(30)
    print("\nPrecision diff for 50x50 matrix")
    precision_dif(50)
    

main()