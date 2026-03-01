import numpy as np
from matplotlib import pyplot as plt 
import time

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x ** 2 - 0.1 / np.exp(-x) + 0.5 * np.cos(2 * x) - 2


# значения по оси абсцисс [-5; 5] с шагом 0.1
coord_x = np.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x) # количество точек
# шаг обучения для каждого параметра w0, w1, w2, w3, w4
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01])
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления EMA

# матрица признаков
X = np.array([[1, x_i, x_i ** 2, np.cos(2 * x_i), np.sin(2 * x_i)]
               for x_i in coord_x])
y = coord_y # целевые значения

Qe = np.mean((X @ w - y) ** 2) # начальное значение функционала качества
np.random.seed(0)

plt.ion() # интерактивный режим графика
fig, ax = plt.subplots()
ax.grid()
ax.plot(coord_x, coord_y, label='func')
approx = ax.plot(coord_x, X @ w, c="red", label='approx')[0]
ax.legend(loc='best')

for i in range(N):
    k = np.random.randint(0, sz) # индекс случайного образа
    L = (X[k] @ w - y[k]) ** 2 # функция потерь для случайного образа
    Qe = lm * L + (1 - lm) * Qe # показатель качества
    # производная функции потерь для случайного образа
    grad_L = 2 * (X[k] * (X[k] @ w - y[k]))
    w = w - eta * grad_L

    approx.set_ydata(X @ w)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.001)

Q = np.mean((X @ w - y) ** 2)

plt.ioff()
plt.show()