import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
x = np.arange(-1, 1, 0.1) # аргумент [-1; 1] с шагом 0,1

model_a = lambda xx, ww: (ww[0] + ww[1] * xx) # модель
# вектор целевых значений с учетом гауссовского шума
y = -5.2 + 0.7 * x + np.random.normal(0, 0.1, len(x)) 

# обучающая выборка, которая включает в себя значения x и столбец
# с 1 (т.к. f0(x) = 1, т.е. коэф. w0 умножается на 1)
X = np.array([[1, xx] for xx in x])

# альтернативное определение обучающей выборки X через 
# единичную матрицу и метод vstack
# E = np.ones((1, len(x))) # единичная матрица
# X = np.vstack((E, x)).T 

w = (np.linalg.inv(X.T @ X)) @ X.T @ y

plt.plot(x, model_a(x, w), marker='s') # линия регрессии
plt.scatter(x, y, color='r') # диаграмма рассеяния с целевыми значениями

plt.title('Простая линейная модель')
plt.xlabel('Значения x')
plt.ylabel('Значения y')
plt.grid()
plt.show()