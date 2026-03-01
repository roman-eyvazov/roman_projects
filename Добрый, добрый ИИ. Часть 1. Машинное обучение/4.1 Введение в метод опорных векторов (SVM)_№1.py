import numpy as np
from matplotlib import pyplot as plt
import time

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.02 * np.exp(-x) - 0.2 * np.sin(3 * x) + 0.5 * np.cos(2 * x) - 7


# производная функции потерь в векторно-матричном виде
def dL(x, w, y):
    return 2 * np.dot(x, x @ w - y)


coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x) # количество значений функций (точек)
# шаг обучения для каждого параметра w0, w1, w2, w3, w4
eta = np.array([0.01, 1e-3, 1e-4, 1e-5, 1e-6])
w = np.array([0., 0., 0., 0., 0.]) # начальные значения весов
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления EMA

# матрица признаков
X = coord_x.reshape(-1, 1) ** np.arange(5)
y = coord_y

Qe = np.mean((X @ w - y) ** 2) # начальное значение среднего эмпирического риска
np.random.seed(0)

plt.ion()
fig, ax = plt.subplots()
ax.grid()
ax.plot(coord_x, coord_y, label='func')
approx = ax.plot(coord_x, X @ w, c="red", label='approx')[0]
ax.legend(loc='best')

for i in range(N):
    k = np.random.randint(0, sz - 1)
    Qe = lm * ((X[k] @ w - y[k]) ** 2) + (1 - lm) * Qe
    w = w - eta * dL(X[k], w, y[k])

    approx.set_ydata(X @ w)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.001)

Q = np.mean((X @ w - y) ** 2)

plt.ioff()
plt.show()