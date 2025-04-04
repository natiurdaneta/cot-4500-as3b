import numpy as np

# Question 1: Gaussian Elimination with Backward Substitution
def gaussian_elimination(A):
    n = len(A)
    for i in range(n):
        pivot = A[i][i]
        for j in range(i + 1, n):
            factor = A[j][i] / pivot
            A[j] -= factor * A[i]
    return A

def backward_substitution(A):
    n = len(A)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = A[i][-1] / A[i][i]
        for j in range(i - 1, -1, -1):
            A[j][-1] -= A[j][i] * x[i]
    return x

# Question 2: LU Factorization
def lu_factorization(A):
    n = len(A)
    L = np.eye(n)
    U = A.copy()
    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            U[j] -= factor * U[i]
    return L, U

# Question 3: Check if Matrix is Diagonally Dominant
def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - np.abs(A[i][i])
        if np.abs(A[i][i]) <= row_sum:
            return False
    return True

# Question 4: Check if Matrix is Positive Definite
def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Main Execution
if __name__ == "__main__":
    # Question 1
    A1 = np.array([[2, -1, 1, 6],
                   [1, 3, 1, 0],
                   [-1, 5, 4, -3]], dtype=float)
    gaussian_elimination(A1)
    x = backward_substitution(A1)
    print("Question 1: Solution to the linear system:", x)

    # Question 2
    A2 = np.array([[1, 1, 0, 3],
                   [2, 1, -1, 1],
                   [3, -1, -1, 2],
                   [-1, 2, 3, -1]], dtype=float)
    L, U = lu_factorization(A2)
    det = np.prod(np.diag(U))
    print("Question 2a: Determinant:", det)
    print("Question 2b: L Matrix:\n", L)
    print("Question 2c: U Matrix:\n", U)

    # Question 3
    A3 = np.array([[9, 0, 5, 2, 1],
                   [3, 9, 1, 2, 1],
                   [0, 1, 7, 2, 3],
                   [4, 2, 3, 12, 2],
                   [3, 2, 4, 0, 8]], dtype=float)
    print("Question 3: Diagonally Dominant?", is_diagonally_dominant(A3))

    # Question 4
    A4 = np.array([[2, 2, 1],
                   [2, 3, 0],
                   [1, 0, 2]], dtype=float)
    print("Question 4: Positive Definite?", is_positive_definite(A4))
