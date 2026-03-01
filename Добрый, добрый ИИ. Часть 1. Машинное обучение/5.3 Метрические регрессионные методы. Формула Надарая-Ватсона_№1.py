import numpy as np
import matplotlib.pyplot as plt

# координаты четырех точек
x = np.array([0, 1, 2, 3])
y = np.array([0.5, 0.8, 0.6, 0.2])

# множество точек для промежуточного восстановления функции
x_est = np.arange(0, 3.1, 0.1)

h = 1 # ширина окна
metric = lambda xx, x_i: abs(xx - x_i) / h # манхэттенская метрика
# треугольное ядро
triangle = lambda xx, x_i: abs(1 - metric(xx, x_i)) * \
                           (abs(metric(xx, x_i)) <= 1)
# по идее, должно быть (1 - abs(r)) - см. конспекты, но принимаю, как 
# написано в условии
# гауссовское ядро
gauss = lambda xx, x_i: 1 / np.sqrt(2 * np.pi) * \
                        np.exp(-metric(xx, x_i) ** 2 / 2)

K = np.array([[triangle(xx, x_i) for x_i in x] for xx in x_est]) # значение ядра
# выход модели по формуле Надарая-Ватсона
y_est = np.sum(y * K, axis=1) / np.sum(K, axis=1)

print(y_est) # на выходе столько же точек, сколько в x_est

# построение графиков
plt.figure(figsize=(14, 8))
plot_number = 0

for kernel in [triangle, gauss]:
    for h in [0.1, 0.5, 1, 5]:
        # значение ядра
        K = np.array([[kernel(xx, x_i) for x_i in x] for xx in x_est])
        # выход модели по формуле Надарая-Ватсона
        y_est = np.sum(y * K, axis=1) / np.sum(K, axis=1)

        plot_number += 1
        plt.subplot(2, 4, plot_number)

        plt.scatter(x, y, color='black', s=10, label='points')
        plt.plot(x_est, y_est, color='red', label='approx')
        plt.title(f'''{'Треугольное' if plot_number < 5 else 'Гауссовское'}
ядро с h = {h}''')
        plt.legend(loc='best')
        plt.grid()

plt.tight_layout()
plt.show()