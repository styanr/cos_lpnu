import matplotlib.pyplot as plt
from fourier import *


def avg_abs_errors(x_values, y_values, approx):
    fourier_errors = []
    least_squares_errors = []

    least_squares_approx = np.polyval(approx[::-1], x_values)

    for i in range(len(x_values)):
        fourier_errors.append(
            abs(y_values[i] - fourier_point(x_values[i], N, a, b)))
        least_squares_errors.append(abs(y_values[i] - least_squares_approx[i]))

    return mean(least_squares_errors), mean(fourier_errors)


y = [2.88, 2.98, 3.24, 3.42, 3.57, 3.79]
x = np.linspace(-math.pi + (2 * math.pi / (len(y) + 1)),
                math.pi, len(y), endpoint=False)
x_fourier = np.linspace(-math.pi, math.pi, 1000)

approx = least_squares_polyfit(x, y, 2)
least_squares_approx = np.polyval(approx[::-1], x_fourier)

N = 4

a, b = fourier_coefficients(x, y, N)

fourier_approx = fourier_range(x_fourier, N, a, b)

print(
    f'Average Error (Least Squares): {avg_abs_errors(x, y, approx)[0]}')
print(f'Average Error (Fourier Series): {avg_abs_errors(x, y, approx)[1]}')
print(
    f'Polynom: {round(approx[2], 5)}x^2 + {round(approx[1], 5)}x + {round(approx[0], 5)}')

plt.plot(x_fourier, fourier_approx, label=f'Fourier Series (N={N})')
plt.plot(x, y, label='Original Data', marker='o', linestyle='None')
plt.plot(x_fourier, least_squares_approx, label='Least Squares Polynomial Fit')
plt.legend()
plt.title('Fourier Series Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
