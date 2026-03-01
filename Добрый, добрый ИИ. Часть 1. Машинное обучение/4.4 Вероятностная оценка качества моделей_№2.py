import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# экспоненциальная функция потерь
def loss(w, x, y):
    M = np.dot(w, x.T) * y
    return np.exp(-M)


# производная экспоненциальной функции потерь по вектору w
def df(w, x, y):
    # таким образом можно получить матрицу нужного размера
    return (-loss(w, x, y) * y) @ x


np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.4
D1 = 2.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 3.0
mean2 = [2, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

# добавляется 1 как отдельный признак
data_x = np.array([[1, x[0], x[1]] for x in np.hstack([x1, x2]).T])
data_y = np.hstack([np.ones(N) * (-1), np.ones(N)])

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, 
                                random_state=123, test_size=0.3, shuffle=True)

n_train = len(X_train) # размер обучающей выборки
w = [0.0, 0.0, 0.0] # начальные весовые коэффициенты
nt = np.array([0.5, 0.01, 0.01]) # шаг обучения для каждого параметра w0, w1, w2
N = 500 # число итераций алгоритма SGD
batch_size = 10 # размер мини-батча (величина K = 10)

for i in range(N):
    # индекс случайного образа
    k = np.random.randint(0, n_train - batch_size - 1)
    index = range(k, k + batch_size) # берем выборку из batch_size значений
    # обратить внимание, что делим на batch_size
    grad_Q = 1 / batch_size * df(w, X_train[index], y_train[index])
    w = w - nt * grad_Q

mrgs = np.sort(X_test @ w * y_test) # сортировка по условию задачи
acc = np.mean(mrgs > 0) # метрика accuracy

print(mrgs)
print(acc)

# построение графика - обратить внимание, что в данном случае в data_x 
# искусственно добавлена 1 как признак, поэтому берем столбцы 1 и 2
x_plot = np.linspace(min(X_test[:, 1]), max(X_test[:, 2]), 1000)
y_plot = - w[1] / w[2] * x_plot - w[0] / w[2] # уравнение разделяющей прямой

plt.plot(x_plot, y_plot, linestyle='--', color='green')
plt.scatter(X_test[y_test == 1][:, 1], X_test[y_test == 1][:, 2], label='+1')
plt.scatter(X_test[y_test == -1][:, 1], X_test[y_test == -1][:, 2], label='-1')
# ошибочные классификации
plt.scatter(X_test[X_test @ w * y_test < 0][:, 1], 
            X_test[X_test @ w * y_test < 0][:, 2], color='red',
            label='mistakes')
plt.title('Модель бинарной классификации')
plt.xlabel('Значения x1')
plt.ylabel('Значения x2')

plt.legend(loc='best')
plt.grid()
plt.show()