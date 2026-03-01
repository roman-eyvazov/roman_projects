import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def func(x):
    return np.sin(0.5 * x) + 0.2 * np.cos(2 * x) - 0.1 * np.sin(4 * x) + 3


# обучающая выборка, можно было бы создать такой же массив (100, 1) 
# через reshape
coord_x = np.expand_dims(np.arange(-4.0, 6.0, 0.1), axis=1)
coord_y = func(coord_x).ravel() # ravel делает массив одномерным

# используем каждый третий отсчет
X_train = coord_x[::3]
y_train = coord_y[::3]

svr = svm.SVR(kernel='rbf') # SVR с нелинейным ядром
svr.fit(X_train, y_train)
predict = svr.predict(coord_x) # делаем прогноз

Q = np.mean((predict - coord_y) ** 2) # функционал качества

print(Q)

# построение графика
plt.plot(coord_x, func(coord_x), label='func')
plt.plot(coord_x, predict, linestyle='-.', label='approx')

plt.legend(loc='best')
plt.grid()
plt.show()