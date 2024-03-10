import matplotlib.pyplot as plt
from fourier import *


def avg_abs_error(y, y_hat):
    return mean([abs(y[i] - y_hat[i]) for i in range(len(y))])


y = [2.88, 2.98, 3.24, 3.42, 3.57, 3.79]
x = np.linspace(-math.pi, math.pi, len(y), endpoint=False)
x_fourier = np.linspace(-math.pi, math.pi, 1000)

approx = least_squares_polyfit(x, y, 2)
least_squares_approx = np.polyval(approx[::-1], x_fourier)

N = 5

a, b = fourier_coefficients(x, y, N)

fourier_approx = fourier_range(x_fourier, N, a, b)

print(
    f'Average Error (Least Squares): {avg_abs_error(y, least_squares_approx)}')
print(f'Average Error (Fourier Series): {avg_abs_error(y, fourier_approx)}')
print(f'Polynom: {approx[2]}x^2 + {approx[1]}x + {approx[0]}')

plt.plot(x_fourier, fourier_approx, label=f'Fourier Series (N={N})')
plt.plot(x, y, label='Original Data')
plt.plot(x_fourier, least_squares_approx, label='Least Squares Polynomial Fit')
plt.legend()
plt.title('Fourier Series Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
