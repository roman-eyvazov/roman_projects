import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = -0.2
D1 = 3.0
mean1 = [1, -5]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-1, -2]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 1000
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * (-1), np.ones(N2)])

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, 
								random_state=123, test_size=0.5, shuffle=True)

clf = svm.SVC(kernel='linear') # SVM с линейным ядром
clf.fit(X_train, y_train) # нахождение вектора w по обучающей выборке

w12 = clf.coef_[0] # коэффициенты w1, w2 разделяющей гиперплоскости
intercept = clf.intercept_[0] # коэффициент w0 разделяющей гиперплоскости
w = np.hstack((intercept, w12)) # сформируем массив всех коэффициентов с w0

print(w)

FPR = [] # список для сохранения значений FPR
TPR = [] # список для сохранения значений TPR

# значения параметра t для построения ROC-кривой
for t in np.arange(5.7, -7.8, -0.1):
	# т.к. нужно выдать прогноз при заданном t, формула прописана вручную
	predict = np.sign(X_test @ w12 + intercept - t) 
	TP = np.sum((predict == +1) & (predict == y_test)) # TP для данного t
	TN = np.sum((predict == -1) & (predict == y_test)) # TN для данного t
	FP = np.sum((predict == +1) & (predict != y_test)) # FP для данного t
	FN = np.sum((predict == -1) & (predict != y_test)) # FN для данного t
	FPR_t = FP / (FP + TN) # FPR для данного t
	TPR_t = TP / (TP + FN) # TPR для данного t

	FPR.append(FPR_t)
	TPR.append(TPR_t)

# построение ROC-кривой
# значения x для прямой, которая задает случайную модель
x = np.linspace(0, 1, 100)
plt.plot(x, x, linestyle='--', label='Случайная модель')
plt.plot(FPR, TPR, label='ROC-кривая')
plt.title('ROC-кривая')
plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend(loc='best')
plt.grid()
plt.show()