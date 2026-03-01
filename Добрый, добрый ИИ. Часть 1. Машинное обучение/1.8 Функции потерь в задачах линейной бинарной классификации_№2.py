import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 0.5 * x ** 2 - 0.1 * 1 / np.exp(-x) + 0.5 * np.cos(2 * x) - 2


coord_x = np.arange(-5.0, 5.0, 0.1) # значения отсчетов по оси абсцисс
coord_y = func(coord_x) # значения функции по оси ординат

w = np.array([-1.59, -0.69, 0.278, 0.497, -0.106]) # задано изначально
# матрица признаков
X = np.array([[1, x_i, x_i ** 2, np.cos(2 * x_i), np.sin(2 * x_i)]
               for x_i in coord_x])
Q = np.mean(abs(X @ w - coord_y))

print(Q)

# построение графика
plt.plot(coord_x, coord_y, linestyle='-', lw=2, label='func')
plt.plot(coord_x, X @ w, linestyle='-.', lw=1, label='approx')

plt.legend(loc='best')
plt.grid()
plt.show()