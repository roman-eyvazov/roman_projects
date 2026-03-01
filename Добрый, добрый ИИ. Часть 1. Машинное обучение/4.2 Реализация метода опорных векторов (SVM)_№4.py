import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = 0.6
D1 = 3.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.7
D2 = 2.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

r3 = 0.5
D3 = 1.0
mean3 = [1, 2]
V3 = [[D3, D3 * r3], [D3 * r3, D3]]

# моделирование обучающей выборки
N = 500
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T
x3 = np.random.multivariate_normal(mean3, V3, N).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y,
								  random_state=123, test_size=0.3, shuffle=True)

clf = svm.SVC(kernel='linear') # SVM с линейным ядром
clf.fit(X_train, y_train)

w1 = clf.coef_[0] # коэффициенты для 1-й разделяющей гиперплоскости
w2 = clf.coef_[1] # коэффициенты для 2-й разделяющей гиперплоскости
w3 = clf.coef_[2] # коэффициенты для 3-й разделяющей гиперплоскости
w01 = clf.intercept_[0] # коэффициент w0 для 1-й гиперплоскости
w02 = clf.intercept_[1] # коэффициент w0 для 2-й гиперплоскости
w03 = clf.intercept_[2] # коэффициент w0 для 3-й гиперплоскости

# сформируем массивы со всеми коэффициентами - в задании проверяется в 
# таком виде
w1 = np.hstack((w01, w1))
w2 = np.hstack((w02, w2))
w3 = np.hstack((w03, w3))

# построение прогнозов меток классов для всей выборки (стратегия one-vs-rest)
predict = clf.predict(X_test)
Q = (predict != y_test).sum() # количество неверных классификаций

print(Q)

# построение графика
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], label='0')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label='1')
plt.scatter(X_test[y_test == 2][:, 0], X_test[y_test == 2][:, 1], label='2')
plt.scatter(X_test[predict != y_test][:, 0], X_test[predict != y_test][:, 1],
			color='red', label='mistakes')

plt.grid()
plt.legend(loc='best')
plt.show()