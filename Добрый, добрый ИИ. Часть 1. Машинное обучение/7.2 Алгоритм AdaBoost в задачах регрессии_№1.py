import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

x = np.arange(-3, 3, 0.1).reshape(-1, 1)
y = 2 * np.cos(x) + 0.5 * np.sin(2 * x) - 0.2 * np.sin(4 * x)

T = 6 # количество моделей
max_depth = 3 # максимальная глубина деревьев
models = [] # пустой список под полученные модели
s = np.array(y.ravel()) # начальная инициализация остатков

for i in range(T):
    # создаем и обучаем решающее дерево
    models.append(DecisionTreeRegressor(max_depth=max_depth))
    models[i].fit(x, s) # обучаем дерево на остатках

    s -= models[i].predict(x) # пересчитываем остатки

# восстанавливаем исходный сигнал по набору полученных деревьев
yy = models[0].predict(x)
for j in range(1, T):
    yy += models[j].predict(x)

# показатель качества (нужно привести yy и y к одной размерности)
QT = np.mean((yy - y.ravel()) ** 2)

print(QT)
# можно было бы сразу найти QT без расчета yy следующим образом
# QT = np.mean(s ** 2)

# построение графика
plt.plot(x, y, label='func')
plt.plot(x, yy, label='approx')
plt.plot(x, s, label='residuals') # остаточный сигнал

plt.legend(loc='best')
plt.grid()
plt.show()