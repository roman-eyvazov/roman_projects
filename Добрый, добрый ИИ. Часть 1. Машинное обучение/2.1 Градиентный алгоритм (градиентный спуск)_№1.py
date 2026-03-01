import numpy as np
import matplotlib.pyplot as plt
import time

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.1 * x ** 3


def df(x):
    return 0.5 + 0.4 * x - 0.3 * x ** 2


coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс
coord_y = func(coord_x) # значения по оси ординат (значения функции)

lm = 0.01 # шаг сходимости
x = -4 # начальное значение x
N = 200 # число итераций

plt.ion() # включение интерактивного режима отображения графиков
fig, ax = plt.subplots() # создание фигуры и осей для графика
ax.grid() # отображение сетки на графике
ax.plot(coord_x, coord_y) # отображение функции
# отображение начальной точки красным цветом
point = ax.scatter(x, func(x), c="red")

for i in range(N):
    x = x - lm * df(x)

    point.set_offsets([x, func(x)]) # отображение нового положения точки
    # перерисовка графика и задержка на 10 мс
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

print(x) # точка минимума

plt.ioff() # выключение интерактивного режима отображения графиков
ax.scatter(x, func(x), c="green") # точка минимума на графике
plt.show()