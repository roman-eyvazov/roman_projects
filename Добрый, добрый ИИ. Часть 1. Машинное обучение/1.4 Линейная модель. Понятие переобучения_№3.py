import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
x = np.arange(-1.0, 1.0, 0.1) # аргумент [-1; 1] с шагом 0,1

# модель
model_a = lambda xx, ww: (ww[0] + ww[1] * xx + ww[2] * xx ** 2
 						  + ww[3] * xx ** 3)
# вектор целевых значений
y = np.sin(x * 5) + 2 * x + np.random.normal(0, 0.1, len(x))

X = np.array([[1, xx, xx ** 2, xx ** 3] for xx in x]) # обучающая выборка

w = (np.linalg.inv(X.T @ X)) @ X.T @ y.reshape(-1, 1)

plt.plot(x, model_a(x, w), marker='s') # линия регрессии
plt.scatter(x, y, color='r') # диаграмма рассеяния с целевыми значениями

plt.title('Линейная модель')
plt.xlabel('Значения x')
plt.ylabel('Значения y')
plt.grid()
plt.show()