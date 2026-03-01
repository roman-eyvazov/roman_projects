import numpy as np
from matplotlib import pyplot as plt
import time

def func(x):
    return 2 * x + 0.1 * x ** 3 + 2 * np.cos(3 * x)


def df(x):
    return 2 + 0.3 * x ** 2 - 6 * np.sin(3 * x)


eta = 0.5 # шаг сходимости
x = 4 # начальное значение x для RMSProp
x_gr = 4 # начальное значение x для обычного градиентного спуска
N = 200 # число итераций градиентного алгоритма
alpha = 0.8
G = 0 # начальное значение G
e = 0.01
lm = 0.5 # шаг для обычного градиентного спуска

coord_x = np.linspace(-10, 10, 100) # массив x для построения графика
coord_y = func(coord_x) # массив y для построения графика

plt.ion() # включение интерактивного режима отображения графиков
fig, ax = plt.subplots()
ax.grid()
ax.plot(coord_x, coord_y)
point = ax.scatter(x, func(x), c="red", label='RMSProp')
point_gd = ax.scatter(x_gr, func(x_gr), c="blue", label='Градиентный спуск')
ax.legend()

for i in range(N):
    G = alpha * G + (1 - alpha) * df(x) * df(x) # RMSProp
    x = x - eta * df(x) / (np.sqrt(G) + e)

    x_gr = x_gr - lm * df(x) # обычный градиентный спуск

    # отображение нового положения точки для RMSProp
    point.set_offsets([x, func(x)])
    # отображение нового положения точки для обычного градиентного спуска
    point_gd.set_offsets([x_gr, func(x_gr)])
    # перерисовка графика и задержка на 20 мс
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

print(x) # точка минимума для RMSProp
print(x_gr) # точка минимума для обычного градиентного спуска

plt.ioff() # выключение интерактивного режима отображения графиков
ax.scatter(x, func(x), c="red")
ax.scatter(x_gr, func(x_gr), c="blue")
plt.show()