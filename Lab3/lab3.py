import math as m
import numpy as np
import scipy.integrate as integrate
from statistics import mean


def least_squares_polyfit(x, y, degree=2):
    if len(x) != len(y):
        raise ValueError("Input arrays x and y must have the same length.")

    b = np.array(y)
    x_values = np.array(x)

    A = np.fliplr(np.vander(x_values, degree + 1, increasing=True))

    coefficients = np.linalg.lstsq(A, b.T, rcond=None)[0]

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
