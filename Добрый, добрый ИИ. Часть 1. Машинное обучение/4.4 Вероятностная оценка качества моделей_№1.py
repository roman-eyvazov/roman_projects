import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = 0.3
D1 = 3.0
mean1 = [2, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 5.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N) * (-1), np.ones(N)])

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, 
							random_state=123, test_size=0.4, shuffle=True)

clf = svm.SVC(kernel='linear') # SVM с линейным ядром
clf.fit(X_train, y_train)

w12 = clf.coef_[0] # коэффициенты w1, w2 разделяющей гиперплоскости
intercept = clf.intercept_[0] # коэффициент w0 разделяющей гиперплоскости
# сформируем массив всех коэффициентов с учетом w0
w = np.hstack((intercept, w12))

predict = clf.predict(X_test) # для бинарной классификации возвращает +1 или -1
acc = np.mean(predict == y_test) # показатель accuracy

print(acc)
# также можно эту рассчитать метрику с помощью метода класса
print(clf.score(X_test, y_test))

# построение графика
plt.figure(figsize=(10, 6))
plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], label='-1')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label='+1')
plt.scatter(X_test[predict != y_test][:, 0], X_test[predict != y_test][:, 1], 
			color='red', label='mistakes') # неверные классификации

# формирование разделяющей плоскости
x_min, x_max = X_test[:, 0].min(), X_test[:, 0].max()
y_min, y_max = X_test[:, 1].min(), X_test[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))

# вычисляем значения функции решения Z на сетке xx, yy с помощью метода 
# decision_function модели SVM
# np.c_ - для конкатенации массивов по второй оси
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# визуализация разделяющей плоскости и контуров
plt.contour(xx, yy, Z, cmap=plt.cm.Set1, alpha=0.8)
plt.legend(loc='best')
plt.title('SVM with Linear kernel')
plt.grid()
plt.show()