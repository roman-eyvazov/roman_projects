import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return 0.1 * x - np.cos(x / 2) + 0.4 * np.sin(3 * x) + 5


np.random.seed(0)

x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
y = func(x) + np.random.normal(0, 0.2, len(x)) # значения функции по оси ординат

x_est = x # в данном случае x_est совпадает с x

h = 0.5 # ширина окна
metric = lambda xx, x_i: abs(xx - x_i) / h # манхэттенская метрика
# гауссовское ядро
gauss = lambda xx, x_i: 1 / np.sqrt(2 * np.pi) * \
                        np.exp(-metric(xx, x_i) ** 2 / 2)
# треугольное ядро с правильной формулой
triangle = lambda xx, x_i: (1 - abs(metric(xx, x_i))) * \
                           (abs(metric(xx, x_i)) <= 1)
# прямоугольное ядро
rectangle = lambda xx, x_i: 0.5 * (abs(metric(xx, x_i)) <= 1)

K = np.array([[gauss(xx, x_i) for x_i in x] for xx in x_est]) # значение ядра
# выход модели по формуле Надарая-Ватсона
y_est = np.sum(y * K, axis=1) / np.sum(K, axis=1)

Q = np.mean((y_est - y) ** 2) # показатель качества восстановления

print(Q)

# построение графиков
plt.figure(figsize=(14, 12))
plot_number = 0

for kernel in [gauss, triangle, rectangle]:
    # словарь для вывода названия ядра
    dict1 = {gauss: 'Гауссовское', triangle:'Треугольное',
             rectangle:'Прямоугольное'}
    for h in [0.1, 0.3, 1, 3]:
        # значение ядра
        K = np.array([[kernel(xx, x_i) for x_i in x] for xx in x_est])
        # выход модели по формуле Надарая-Ватсона
        y_est = np.sum(y * K, axis=1) / np.sum(K, axis=1)

        plot_number += 1
        plt.subplot(3, 4, plot_number)

        plt.scatter(x, y, color='black', s=10, label='func')
        plt.plot(x_est, y_est, color='red', label='approx')
        plt.title(f'{dict1[kernel]} ядро с h = {h}')
        plt.legend(loc='best')
        plt.grid()

plt.tight_layout()
plt.show()