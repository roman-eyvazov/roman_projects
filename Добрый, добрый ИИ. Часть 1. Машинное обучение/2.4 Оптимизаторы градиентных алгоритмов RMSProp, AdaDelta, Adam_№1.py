import numpy as np
from matplotlib import pyplot as plt
import time

def func(x):
    return -0.5 * x + 0.2 * x ** 2 - 0.01 * x ** 3 - 0.3 * np.sin(4 * x)


def df(x):
    return -0.5 + 0.4 * x - 0.03 * x ** 2 - 1.2 * np.cos(4 * x)


eta = 0.1 # шаг сходимости
x = -3.5 # начальное значение x для метода импульсов 
x_gr = -3.5 # начальное значение x для обычного градиентного спуска
N = 200 # число итераций градиентного алгоритма
gamma = 0.8
v = 0 # начальное значение v
lm = 0.1 # шаг для обычного градиентного спуска

coord_x = np.linspace(-10, 10, 100) # массив x для построения графика
coord_y = func(coord_x) # массив y для построения графика

plt.ion() # включение интерактивного режима отображения графиков
fig, ax = plt.subplots()
ax.grid()
ax.plot(coord_x, coord_y)
# отображение начальной точки для метода импульсов
point = ax.scatter(x, func(x), c="red", label='Метод импульсов')
# отображение начальной точки для обычного градиентного спуска
point_gd = ax.scatter(x_gr, func(x_gr), c="blue", label='Градиентный спуск')
ax.legend()

for i in range(N):
    v = gamma * v + (1 - gamma) * eta * df(x) # метод импульсов
    x = x - v

    x_gr = x_gr - lm * df(x) # обычный градиентный спуск

    # отображение нового положения точки для метода импульсов
    point.set_offsets([x, func(x)])
    # отображение нового положения точки для обычного градиентного спуска
    point_gd.set_offsets([x_gr, func(x_gr)])
    # перерисовка графика и задержка на 20 мс
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

print(x) # точка минимума для метода импульсов
print(x_gr) # точка минимума для обычного градиентного спуска

plt.ioff() # выключение интерактивного режима отображения графиков
# точка минимума на графике для метода импульсов
ax.scatter(x, func(x), c="red")
# точка минимума на графике для обычного градиентного спуска
ax.scatter(x_gr, func(x_gr), c="blue")
plt.show()