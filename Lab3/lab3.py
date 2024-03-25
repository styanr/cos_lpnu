import math as m
import numpy as np
import scipy.integrate as integrate
from statistics import mean


def least_squares_polyfit(x, y, degree=2):
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


def precompute_coefficients(iterations, coefficients, file=None):
    """
    Precomputes `a_n` and `b_n` coefficients for all iterations.
    """
    a_n = [0] * (iterations + 1)
    b_n = [0] * (iterations + 1)

    for n in range(iterations + 1):
        a_n[n] = (1 / m.pi) * integrate.quad(lambda x: np.polyval(coefficients, x)
                                             * m.cos(n * x), -m.pi, m.pi)[0]
        b_n[n] = (1 / m.pi) * integrate.quad(lambda x: np.polyval(coefficients, x)
                                             * m.sin(n * x), -m.pi, m.pi)[0]
        if file:
            file.write(
                f"a_{n} = {a_n[n]}\tb_{n} = {b_n[n]}\n")

    return a_n, b_n


def fourier_range(x_values, coefficients, iterations=50, file=None):
    """
    The Fourier series approximation. Logs the generated values if a file is given.
    """
    a_n, b_n = precompute_coefficients(iterations, coefficients, file)

    if file:
        file.write(f"Iterations: {iterations:.0f}\n\n")

    fourier_range = [fourier_series(x, a_n, b_n) for x in x_values]

    return fourier_range


def fourier_series(x, a_n, b_n):
    """
    The Fourier series. Logs the a_n and b_n coefficients if a file is given.
    """
    sum = 0
    for n in range(1, len(a_n)):
        sum += a_n[n] * m.cos(n * x) + b_n[n] * m.sin(n * x)

    return a_n[0] / 2 + sum


def avg_abs_errors(x_values, y_values, approx, a, b):
    N = len(a) - 1
    fourier_errors = []
    least_squares_errors = []

    least_squares_approx = np.polyval(approx, x_values)

    for i in range(len(x_values)):
        fourier_errors.append(
            abs(y_values[i] - fourier_series(x_values[i], a, b)))
        least_squares_errors.append(abs(y_values[i] - least_squares_approx[i]))

    return mean(least_squares_errors), mean(fourier_errors)
