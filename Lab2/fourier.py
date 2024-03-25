import math
import numpy as np
from statistics import mean


def a_k(x, y, k):
    n = len(x)
    return (2 / n) * sum([y[i] * math.cos(k * x[i]) for i in range(n)])


def b_k(x, y, k):
    n = len(x)
    return (2 / n) * sum([y[i] * math.sin(k * x[i]) for i in range(n)])


def fourier_coefficients(x, y, N):
    a = [a_k(x, y, k) for k in range(N + 1)]
    b = [b_k(x, y, k) for k in range(N + 1)]

    return a, b


def fourier_range(x, N, a, b):
    n = len(x)
    f = np.zeros(n)
    for i in range(n):
        f[i] = fourier_point(x[i], a, b)
    return f


def fourier_point(x, a, b):
    N = len(a) - 1
    f = a[0] / 2 + sum([a[k] * math.cos(k * x) +
                        b[k] * math.sin(k * x) for k in range(1, N + 1)])
    return f


def least_squares_polyfit(x, y, degree):
    if len(x) != len(y):
        raise ValueError("Input arrays x and y must have the same length.")

    n = len(x)
    m = degree + 1  # Number of coefficients

    # Build the Vandermonde matrix
    A = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            A[i][j] = x[i] ** (m - j - 1)

    # Build the b vector
    b = y.copy()

    # Solve the normal equations: (A^T A) c = A^T b
    AT = [[sum(A[k][j] * A[k][i] for k in range(n))
           for j in range(m)] for i in range(m)]
    ATb = [sum(A[k][i] * b[k] for k in range(n)) for i in range(m)]

    # Gaussian elimination to solve (A^T A) c = A^T b
    for i in range(m):
        pivot = i
        for j in range(i + 1, m):
            if abs(AT[j][i]) > abs(AT[pivot][i]):
                pivot = j

        AT[i], AT[pivot] = AT[pivot], AT[i]
        ATb[i], ATb[pivot] = ATb[pivot], ATb[i]

        for j in range(i + 1, m):
            factor = AT[j][i] / AT[i][i]
            for k in range(i + 1, m):
                AT[j][k] -= factor * AT[i][k]
            ATb[j] -= factor * ATb[i]

    # Back-substitution
    coefficients = [0 for _ in range(m)]
    coefficients[m - 1] = ATb[m - 1] / AT[m - 1][m - 1]
    for i in range(m - 2, -1, -1):
        coefficients[i] = (ATb[i] - sum(AT[i][j] * coefficients[j]
                           for j in range(i + 1, m))) / AT[i][i]

    return coefficients


def avg_abs_errors(x_values, y_values, approx, a, b):
    N = len(a) - 1
    fourier_errors = []
    least_squares_errors = []

    least_squares_approx = np.polyval(approx[::-1], x_values)

    for i in range(len(x_values)):
        fourier_errors.append(
            abs(y_values[i] - fourier_point(x_values[i], a, b)))
        least_squares_errors.append(abs(y_values[i] - least_squares_approx[i]))

    return mean(least_squares_errors), mean(fourier_errors)
