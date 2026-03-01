import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = 0.2
D1 = 3.0
mean1 = [2, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-1, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 2500
N2 = 1500
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * (-1), np.ones(N2)])

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, 
								random_state=123, test_size=0.4, shuffle=True)

clf = svm.SVC(kernel='linear') # SVM с линейным ядром
clf.fit(X_train, y_train) # нахождение вектора w по обучающей выборке
predict = clf.predict(X_test) # возвращает +1 или -1 для бинарной классификации

w12 = clf.coef_[0] # коэффициенты w1, w2 разделяющей гиперплоскости
intercept = clf.intercept_[0] # коэффициент w0 разделяющей гиперплоскости
# сформируем массив всех коэффициентов с учетом w0
w = np.hstack((intercept, w12))

print(w)

TP = np.sum((predict == +1) & (predict == y_test))
TN = np.sum((predict == -1) & (predict == y_test))
FP = np.sum((predict == +1) & (predict != y_test))
FN = np.sum((predict == -1) & (predict != y_test))
precision = TP / (TP + FP) # precision для тестовой выборки
recall = TP / (TP + FN) # recall для тестовой выборки
F = 2 * precision * recall / (precision + recall) # F-мера для тестовой выборки
# Fb-мера при бета = 0.5 для тестовой выборки
Fb = (1 + 0.5 ** 2) * precision * recall / (0.5 ** 2 * precision + recall)

print(F)
print(Fb)

# построение графика - только для тестовой выборки
x_plot = np.linspace(min(X_test[:, 0]), max(X_test[:, 0]), 1000)
y_plot = - w[1] / w[2] * x_plot - w[0] / w[2] # уравнение разделяющей прямой

plt.plot(x_plot, y_plot, linestyle='--', color='green')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label='+1')
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], label='-1')
# ошибочные классификации
plt.scatter(X_test[predict != y_test][:, 0], X_test[predict != y_test][:, 1], 
			color='black', marker='x', label='mistakes')

plt.title('Применение SVM с линейным ядром')
plt.xlabel('Значения x1')
plt.ylabel('Значения x2')

plt.legend(loc='best')
plt.grid()
plt.show()