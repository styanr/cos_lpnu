import math as m
from statistics import mean
import scipy.integrate as integrate


def f(x):
    """The function given in the task."""
    mod = (x + m.pi) % (2 * m.pi) - m.pi
    return -1 if -m.pi < mod <= 0 else 0


def f_2(x):
    mod = (x + m.pi) % (2 * m.pi) - m.pi
    # return -1 if -m.pi < mod <= 0 else 0
    return (mod ** 22) * m.exp((-mod ** 2) / 22)


def precompute_coefficients(iterations, func, file=None):
    """
    Precomputes `a_n` and `b_n` coefficients for all iterations.
    """
    a_n = [0] * (iterations + 1)
    b_n = [0] * (iterations + 1)

    for n in range(iterations + 1):
        a_n[n] = (1 / m.pi) * integrate.quad(lambda x: func(x)
                                             * m.cos(n * x), -m.pi, m.pi)[0]
        b_n[n] = (1 / m.pi) * integrate.quad(lambda x: func(x)
                                             * m.sin(n * x), -m.pi, m.pi)[0]
        if file:
            file.write(
                f"a_{n} = {a_n[n]}\tb_{n} = {b_n[n]}\n")

    return a_n, b_n


def fourier_range(x_values, iterations=50, file=None):
    """
    The Fourier series approximation. Logs the generated values if a file is given.
    """
    a_n, b_n = precompute_coefficients(iterations, f, file)

    if file:
        file.write(f"Iterations: {iterations:.0f}\n\n")

    fourier_range = [fourier_series(x, a_n, b_n) for x in x_values]
    error = absolute_error([f(x) for x in x_values], fourier_range)
    quad_error = quadratic_error(a_n, b_n)

    if file:
        file.write(
            f"Absolute error: {error}\n")

    return fourier_range, error, quad_error


def fourier_range_2(x_values, iterations=50, file=None):
    """
    The Fourier series approximation. Logs the generated values if a file is given.
    """
    a_n, b_n = precompute_coefficients(iterations, f_2, file)

    if file:
        file.write(f"Iterations: {iterations:.0f}\n\n")

    fourier_range = [fourier_series(x, a_n, b_n) for x in x_values]
    error = relative_error([f_2(x) for x in x_values], fourier_range)
    quad_error = quadratic_error(a_n, b_n)

    if file:
        file.write(
            f"Relative error: {error}\n")

    return fourier_range, error, quad_error


def fourier_series(x, a_n, b_n):
    """
    The Fourier series. Logs the a_n and b_n coefficients if a file is given.
    """
    sum = 0
    for n in range(1, len(a_n)):
        sum += a_n[n] * m.cos(n * x) + b_n[n] * m.sin(n * x)

    return a_n[0] / 2 + sum


def absolute_error(F, Fourier):
    """
    The absolute error.
    """
    return mean(abs(f - fourier) for f, fourier in zip(F, Fourier))


def relative_error(F, Fourier):
    """
    The relative error.
    """
    total_error = 0
    count = 0

    for f, fourier in zip(F, Fourier):
        total_error += abs(f - fourier) / max(abs(f), abs(fourier))
        count += 1

    return total_error / count


def quadratic_error(a_n, b_n):
    """
    The quadratic error.
    """
    return 1 / (2 * m.pi) * integrate.quad(lambda x: (f(x)**2), -m.pi, m.pi)[0] - a_n[0]**2 / 4 - 0.5 * sum(a_n[n]**2 + b_n[n]**2 for n in range(1, len(a_n)))
