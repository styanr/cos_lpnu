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


def least_squares_polyfit(x, y, degree=2):
    if len(x) != len(y):
        raise ValueError("Input arrays x and y must have the same length.")

    b = np.array(y)
    x_values = np.array(x)

    A = np.vander(x_values, degree + 1, increasing=True)

    coefficients = np.linalg.lstsq(A, b.T, rcond=None)[0]

    print(coefficients)

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
