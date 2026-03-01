import numpy as np
from matplotlib import pyplot as plt
import time

def func(x):
    return 0.4 * x + 0.1 * np.sin(2 * x) + 0.2 * np.cos(3 * x)


def df(x):
    return 0.4 + 0.2 * np.cos(2 * x) - 0.6 * np.sin(3 * x)


eta = 1 # шаг сходимости
x = 4 # начальное значение x для SGD с импульсом Нестерова
x_gr = 4 # начальное значение x для обычного градиентного спуска
N = 500 # число итераций градиентного алгоритма
gamma = 0.7
v = 0 # начальное значение v
lm = 1 # шаг для обычного градиентного спуска

coord_x = np.linspace(-10, 10, 100) # массив x для построения графика
coord_y = func(coord_x) # массив y для построения графика

plt.ion() # включение интерактивного режима отображения графиков
fig, ax = plt.subplots()
ax.grid()
ax.plot(coord_x, coord_y)
# отображение начальной точки для SGD с импульсом Нестерова
point = ax.scatter(x, func(x), c="red", label='SGD с импульсом Нестерова')
# отображение начальной точки для обычного градиентного спуска
point_gd = ax.scatter(x_gr, func(x_gr), c="blue", label='Градиентный спуск')
ax.legend()

for i in range(N):
    # SGD с импульсом Нестерова
    v = gamma * v + (1 - gamma) * eta * df(x - gamma * v)
    x = x - v

    x_gr = x_gr - lm * df(x) # обычный градиентный спуск

    # отображение нового положения точки для SGD с импульсом Нестерова
    point.set_offsets([x, func(x)])
    # отображение нового положения точки для обычного градиентного спуска
    point_gd.set_offsets([x_gr, func(x_gr)])
    # перерисовка графика и задержка на 20 мс
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

print(x) # точка минимума для SGD с импульсом Нестерова
print(x_gr) # точка минимума для обычного градиентного спуска

plt.ioff() # выключение интерактивного режима отображения графиков
# точка минимума на графике для SGD с импульсом Нестерова
ax.scatter(x, func(x), c="red")
# точка минимума на графике для обычного градиентного спуска
ax.scatter(x_gr, func(x_gr), c="blue")
plt.show()