import matplotlib.pyplot as plt
from fourier import *


y = [2.88, 2.98, 3.24, 3.42, 3.57, 3.79]
x = np.linspace(-math.pi + (2 * math.pi / (len(y) + 1)),
                math.pi, len(y), endpoint=False)
x_fourier = np.linspace(-math.pi, math.pi, 1000)

approx = least_squares_polyfit(x, y, 2)
least_squares_approx = np.polyval(approx[::-1], x_fourier)

N = 6

a, b = fourier_coefficients(x, y, N)

with open('fourier_coefficients.txt', 'w') as f:
    for i in range(len(a)):
        f.write(f'a_{i}: {a[i]:5f}\t\tb_{i}: {b[i]:5f}\n')

fourier_approx = fourier_range(x_fourier, N, a, b)

lstsq_error, fourier_error = avg_abs_errors(x, y, approx, a, b)

print(
    f'Average Error (Least Squares): {lstsq_error}')
print(f'Average Error (Fourier Series): {fourier_error}')
print(
    f'Polynom: {round(approx[2], 5)}x^2 + {round(approx[1], 5)}x + {round(approx[0], 5)}')

with open('errors.txt', 'w') as f:
    f.write(f'Average Error (Least Squares): {lstsq_error}\n')
    f.write(f'Average Error (Fourier Series): {fourier_error}\n')
    f.write(
        f'Polynom: {round(approx[2], 5)}x^2 + {round(approx[1], 5)}x + {round(approx[0], 5)}')

plt.plot(x_fourier, fourier_approx, label=f'Fourier Series (N={N})')
plt.plot(x, y, label='Original Data', marker='o', linestyle='None')
plt.plot(x_fourier, least_squares_approx, label='Least Squares Polynomial Fit')
plt.legend()
plt.title('Fourier Series Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
