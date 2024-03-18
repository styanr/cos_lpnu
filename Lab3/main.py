import math
import numpy as np
import matplotlib.pyplot as plt

from lab3 import *


def main():
    y = [2.73, 2.31, 1.97, 1.76, 1.53, 1.31]
    x = np.linspace(-math.pi + (2 * math.pi / (len(y) + 1)),
                    math.pi, len(y), endpoint=False)

    N = 10

    x_fourier = np.linspace(-math.pi, math.pi, 100)
    y_fourier = fourier_range(x_fourier, least_squares_polyfit(x, y), N)

    lstq_error, fourier_error = avg_abs_errors(x, y, least_squares_polyfit(
        x, y), *precompute_coefficients(N, least_squares_polyfit(x, y)))

    print(f"Least Squares Error: {lstq_error}")
    print(f"Fourier Error: {fourier_error}")
    coefficients = least_squares_polyfit(x, y, 2)
    print(
        f"Polynom: {coefficients[0]:3f}x^2 + {coefficients[1]:3f}x + {coefficients[2]:3f}")

    plt.plot(x, y, 'ro', label="Data Points")
    plt.plot(x, np.polyval(coefficients, x), label="Least Squares Fit")
    plt.plot(x_fourier, y_fourier, label="Fourier Series")
    plt.legend()
    plt.show()


if (__name__ == "__main__"):
    main()
