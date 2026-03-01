import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


def model(w, x):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3


coord_x = np.arange(-4, 6, 0.1)

X_train = coord_x.reshape(-1, 1) ** np.arange(4) # матрица признаков
y_train = func(coord_x) # целевая переменная

# решаем аналитически без теоремы Байеса
w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
Q = np.mean((func(coord_x) - model(w, coord_x)) ** 2)

print(Q)

# построение графика
plt.plot(coord_x, func(coord_x), lw=2, label='func')
plt.plot(coord_x, X_train @ w, linestyle='-.', label='approx')

plt.legend(loc='best')
plt.grid()
plt.show()