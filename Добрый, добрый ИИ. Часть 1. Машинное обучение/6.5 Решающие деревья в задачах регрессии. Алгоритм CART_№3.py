import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

x = np.arange(-2, 3, 0.1).reshape(-1, 1)
y = 0.3 * x ** 2 - 0.2 * x ** 3 - 0.5 * np.sin(4 * x)

clf = tree.DecisionTreeRegressor(max_depth=4)
clf.fit(x, y)
pr_y = clf.predict(x) # делаем прогноз

# показатель качества - нужно y привести к форме вектора-строки для
# корректного расчета
Q = np.mean((pr_y - y.reshape(1, -1)) ** 2)

print(Q)

# построение графика
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(x, y, label='func')
ax[0].plot(x, pr_y, label='DT, max_depth=4')
ax[0].legend(loc='best')
ax[0].grid()

ax[1].plot = tree.plot_tree(clf) # визуализация дерева

plt.show()