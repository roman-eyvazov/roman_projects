import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4 * x) + np.cos(2 * x)

t = 0 # заданный порог

# константные значения в листьях дерева
b1, b2 = np.mean(y[x < t]), np.mean(y[x >= t])
b = np.mean(y) # константное значение для расчета H(R) - начального impurity
# impurity для 2 выборок - мы считаем его как сумму, а не как среднее
HR1, HR2 = np.sum((b1 - y[x < t]) ** 2), np.sum((b2 - y[x >= t]) ** 2)
# начальное impurity для всей выборки
HR = np.sum((b - y) ** 2)
IG = HR - len(y[x < t]) / len(y) * HR1 - len(y[x >= t]) / len(y) * HR2

print(IG)

# построение графика
plt.plot(x, y, label='func')
plt.hlines(y=b1, xmin=min(x), xmax=t, label='b1', color='red')
plt.hlines(y=b, xmin=min(x), xmax=max(x), label='b', color='red')
plt.hlines(y=b2, xmin=t, xmax=max(x), label='b2', color='red')

plt.legend(loc='best')
plt.grid()
plt.show()