import numpy as np
import matplotlib.pyplot as plt

rub_usd = np.array([75, 76, 79, 82, 85, 81, 83, 86, 87, 85, 83, 80, 77, 79, 78,
                    81, 84])

days = 10 # количество прогнозных значений

x = list(range(len(rub_usd))) # значения x для имеющихся дней
# значения x для прогнозных дней
x_est = list(range(len(rub_usd), len(rub_usd) + days))
# список под прогнозные значения - возьмем оттуда только последние 10 дней
predict = rub_usd.tolist()

h = 3 # ширина окна
metric = lambda xx, x_i: abs(xx - x_i) / h # манхэттенская метрика
# гауссовское ядро
gauss = lambda xx, x_i: 1 / np.sqrt(2 * np.pi) * \
                        np.exp(-metric(xx, x_i) ** 2 / 2)

# при построении очередного прогноза нужно использовать предыдущие
# прогнозные значения
for xx in x_est:
    K = np.array([gauss(xx, x_i) for x_i in x]) # значение ядра
    y_est = np.sum(predict * K) / np.sum(K) # прогнозное значение на один день

    # добавляем в x очередной день, на который сделан прогноз
    x.append(x[-1] + 1)
    # добавляем в predict очередной прогноз, чтобы учесть его в дальнейшем
    predict.append(y_est)

predict = predict[-10:] # берем последние 10 значений
print(predict)

# построение графика
plt.plot(x[:len(rub_usd)], rub_usd, color='red', label='Имеющиеся значения')
plt.plot(x_est, predict, color='blue', label='Прогноз')
plt.title(f'Прогноз курса USD/RUB с гауссовским ядром и h = {h}')

plt.legend(loc='best')
plt.grid()
plt.show()