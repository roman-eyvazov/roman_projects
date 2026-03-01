import numpy as np
import matplotlib.pyplot as plt
import time

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.1 * x ** 2 - np.sin(x) + 5


# значения по оси абсцисс [-5; 5] с шагом 0.1
coord_x = np.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x) # количество точек
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для весов w
w = np.array([0., 0., 0., 0.]) # начальные значения весов модели
N = 200 # число итераций градиентного алгоритма

X = coord_x.reshape(-1, 1) ** np.arange(4) # матрица признаков
# альтернативный вариант
# X = np.array([[x_i ** i for i in range(4)] for x_i in coord_x])
y = coord_y # целевая переменная

plt.ion()
fig, ax = plt.subplots()
ax.grid()
ax.plot(coord_x, coord_y, label='func')
# график функции и первоначальная аппроксимация
approx = ax.plot(coord_x, X @ w, c="red", label='approx')[0]
ax.legend(loc='best')

for i in range(N):
    grad = 2 / sz * X.T @ (X @ w - y)
    w = w - eta * grad

    approx.set_ydata(X @ w.T)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

Q = np.mean((X @ w - y) ** 2)

print(Q)

plt.ioff()
plt.show()