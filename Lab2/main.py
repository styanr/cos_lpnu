import matplotlib.pyplot as plt
from fourier import *

y = [2.88, 2.98, 3.24, 3.42, 3.57, 3.79]
x = np.linspace(-math.pi, math.pi, len(y), endpoint=False)
x_fourier = np.linspace(-2 * math.pi, 2 * math.pi, 1000)

# Number of Fourier terms
N = 5

# Compute Fourier coefficients
a, b = fourier_coefficients(x, y, N)

# Generate Fourier series approximation
f_approx = fourier_range(x_fourier, N, a, b)

# Plotting the original data and Fourier series approximation
plt.plot(x_fourier, f_approx, label=f'Fourier Series (N={N})')
plt.plot(x, y, label='Original Data')
plt.legend()
plt.title('Fourier Series Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
