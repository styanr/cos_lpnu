import math as m
from statistics import mean
import scipy.integrate as integrate


def f(x):
    """The function given in the task."""
    mod = (x + m.pi) % (2 * m.pi) - m.pi

    if -m.pi < mod <= 0:
        return -1
    elif 0 < mod <= m.pi:
        return 0


def a_n(n):
    """The a_n coefficient."""
    return (1 / m.pi) * integrate.quad(lambda x: f(x) * m.cos(n * x), -m.pi, m.pi)[0]


def b_n(n):
    """The b_n coefficient."""
    return (1 / m.pi) * integrate.quad(lambda x: f(x) * m.sin(n * x), -m.pi, m.pi)[0]


def fourier_range(x_values, iterations=50, file=None):
    """The Fourier series approximation. Logs the generated values if a file is given."""
    if file:
        file.write(f"Iterations: {iterations:.0f}\n\n")

    fourier_range = [fourier_series(x, iterations, file) for x in x_values]

    if file:
        file.write(
            f"Absolute error: {absolute_error([f(x) for x in x_values], fourier_range)}\n")

    return fourier_range


def fourier_series(x, iterations=50, file=None):
    """The Fourier series. Logs the a_n and b_n coefficients if a file is given."""
    a_0 = a_n(0)
    sum = 0
    n = 1

    while n <= iterations:
        sum += a_n(n) * m.cos(n * x) + b_n(n) * m.sin(n * x)
        if file:
            file.write(f"a[{n:2d}]: {a_n(n)}\tb[{n:2d}]: {b_n(n)}\n")

        n += 1

    if file:
        file.write("\n")

    return a_0 / 2 + sum


def absolute_error(F, Fourier):
    """The absolute error."""
    return mean(abs(f - fourier) for f, fourier in zip(F, Fourier))
