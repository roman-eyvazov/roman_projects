import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# исходные параметры распределений трех классов
r1 = 0.7
D1 = 3.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

r3 = 0.3
D3 = 1.0
mean3 = [1, 2]
V3 = [[D3, D3 * r3], [D3 * r3, D3]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T
x3 = np.random.multivariate_normal(mean3, V3, N).T

X_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2]) # классы 0, 1 и 2

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

a1, a2, a3 = (x1.T - mm1).T, (x2.T - mm2).T, (x3.T - mm3).T
VV1 = 1 / N * a1 @ a1.T
VV2 = 1 / N * a2 @ a2.T
VV3 = 1 / N * a3 @ a3.T

# параметры для гауссовского байесовского классификатора
Py1, Py2, Py3 = 0.2, 0.5, 0.3
L1, L2, L3 = 1, 1, 1

b = lambda x, m, v, L, P: np.log(L * P) - 0.5 * (x - m) @ np.linalg.inv(v) \
                          @ (x - m).T - 0.5 * np.log(np.linalg.det(v))
predict = np.zeros(len(X_train))

for i in range(len(X_train)):
    predict[i] = np.argmax([b(X_train[i], mm1, VV1, L1, Py1), 
                            b(X_train[i], mm2, VV2, L2, Py2), 
                            b(X_train[i], mm3, VV3, L3, Py3)])

Q = np.sum(predict != y_train)

print(Q)

# построение графика
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='0')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='1')
plt.scatter(X_train[y_train == 2][:, 0], X_train[y_train == 2][:, 1], label='2')
plt.scatter(X_train[predict != y_train][:, 0], 
            X_train[predict != y_train][:, 1], color='red', label='mistakes')

plt.grid()
plt.legend(loc='best')
plt.show()