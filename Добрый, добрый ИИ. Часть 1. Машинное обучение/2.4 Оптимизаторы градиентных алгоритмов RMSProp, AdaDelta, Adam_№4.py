import numpy as np
from matplotlib import pyplot as plt 
import time

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return -0.7 * x - 0.2 * x ** 2 + 0.05 * x ** 3 - 0.2 * np.cos(3 * x) + 2


# значения по оси абсцисс [-4; 6] с шагом 0.1
coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x) # количество точек
# шаг обучения для каждого параметра w0, w1, w2, w3
eta = np.array([0.1, 0.01, 0.001, 0.0001])
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления EMA
batch_size = 20 # размер мини-батча (величина K = 20)
gamma = 0.8 # коэффициент гамма для вычисления импульсов Нестерова
v = np.zeros(len(w)) # начальное значение [0, 0, 0, 0]

X = coord_x.reshape(-1, 1) ** np.arange(4) # матрица признаков
y = coord_y # целевые значения

plt.ion()
fig, ax = plt.subplots()
ax.grid()
ax.plot(coord_x, coord_y, label='func')
approx = ax.plot(coord_x, X @ w, c="red", label='approx')[0]
ax.legend(loc='best')

Qe = np.mean((X @ w - y) ** 2) # начальное значение среднего эмпирического риска
np.random.seed(0)

for i in range(N):
    k = np.random.randint(0, sz - batch_size - 1)
    index = k + batch_size
    Q = np.mean((X[k:index] @ w - y[k:index]) ** 2) # эмпирический риск
    Qe = lm * Q + (1 - lm) * Qe # функционал качества
    grad_Q = 2 / batch_size * ((X[k:index] @ (w - gamma * v) -
                                y[k:index]) @ X[k:index])
    v = gamma * v + (1 - gamma) * eta * grad_Q
    w = w - v

    approx.set_ydata(X @ w)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.001)

Q = np.mean((X @ w - y) ** 2)

plt.ioff()
plt.show()