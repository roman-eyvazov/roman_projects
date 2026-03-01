import numpy as np
from matplotlib import pyplot as plt 
import time

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
     return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


coord_x = np.arange(-4.0, 6.0, 0.1) # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x) # количество значений функций (точек)
# шаг обучения для каждого параметра w0, w1, w2, w3
eta = np.array([0.1, 0.01, 0.001, 0.0001])
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления EMA
batch_size = 50 # размер мини-батча (величина K = 50)

# матрица признаков
X = np.array([[x_i ** i for i in range(4)] for x_i in coord_x])
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
    k = np.random.randint(0, sz - batch_size) # индекс случайного образа
    # берем выборку из batch_size значений - не оптимально
    # (лучше брать разные индексы через choice)
    index = k + batch_size 
    Q = np.mean((X[k:index] @ w - y[k:index]) ** 2) # эмпирический риск
    Qe = lm * Q + (1 - lm) * Qe # показатель качества
    # производная функции потерь
    grad_Q = 2 / batch_size * (X[k:index].T @ (X[k:index] @ w - y[k:index]))
    w = w - eta * grad_Q

    approx.set_ydata(X @ w)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

Q = np.mean((X @ w - y) ** 2)

plt.ioff()
plt.show()