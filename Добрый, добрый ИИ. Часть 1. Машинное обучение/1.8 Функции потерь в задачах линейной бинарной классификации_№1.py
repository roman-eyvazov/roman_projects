import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 0.1 * x ** 2 - np.sin(x) + 0.1 * np.cos(x * 5) + 1


coord_x = np.arange(-5.0, 5.0, 0.1) # значения отсчетов по оси абсцисс
coord_y = func(coord_x) # значения функции по оси ординат

w = np.array([1.11, -0.26, 0.061, 0.0226, 0.00178]) # задано изначально
X = coord_x.reshape(-1, 1) ** np.arange(0, 5) # матрица признаков
# альтернативный вариант для X
# X = np.array([[x_i ** i for i in range(5)] for x_i in coord_x])
Q = np.mean((X @ w - coord_y) ** 2)

print(Q)

# построение графика
plt.plot(coord_x, coord_y, linestyle='-', lw=2, label='func')
plt.plot(coord_x, X @ w, linestyle='-.', lw=1, label='approx')

plt.legend(loc='best')
plt.grid()
plt.show()