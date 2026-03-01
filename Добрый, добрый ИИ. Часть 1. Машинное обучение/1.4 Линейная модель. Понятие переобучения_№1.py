import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
x = np.arange(-1, 1, 0.1) # аргумент [-1; 1] с шагом 0,1

size_train = len(x) # размер выборки
w = [0.5, -0.3] # коэффициенты модели
model_a = lambda m_x, m_w: (m_w[1] * m_x + m_w[0]) # модель
loss = lambda ax, y: (ax - y) ** 2 # квадратическая функция потерь

y = model_a(x, w) + np.random.normal(0, 0.1, size_train) # целевые значения

Q = np.mean(loss(model_a(x, w), y)) # средний эмпирический риск
# видно, что Q равен среднеквадратическому отклонению в квадрате
# (задается в np.random.normal) - что абсолютно логично в данном случае
print(Q)

plt.plot(x, model_a(x, w), color='g') # линия регрессии
plt.scatter(x, y, color='r') # диаграмма рассеяния с целевыми значениями

plt.title('Простая линейная модель')
plt.xlabel('Значения x')
plt.ylabel('Значения y')
plt.grid()
plt.show()