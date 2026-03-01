import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

x = np.arange(-3, 3, 0.1)
y = 0.3 * x + np.cos(2 * x) + 0.2 * np.sin(7 * x) # + \
# 	np.random.normal(0.0, 0.1, n_samples)
x = x.reshape(-1, 1)

T = 5  # число деревьев

rf = RandomForestRegressor(max_depth=8, n_estimators=T, random_state=1)
rf.fit(x, y)
pr_y = rf.predict(x) # делаем прогноз

# показатель качества - нужно y привести к форме вектора-строки для
# корректного расчета
Q = np.mean((pr_y - y.reshape(1, -1)) ** 2)

print(Q)

# построение графика
plt.plot(x, y, label='func')
plt.plot(x, pr_y, label='RF, max_depth=8')

plt.legend(loc='best')
plt.grid()
plt.show()