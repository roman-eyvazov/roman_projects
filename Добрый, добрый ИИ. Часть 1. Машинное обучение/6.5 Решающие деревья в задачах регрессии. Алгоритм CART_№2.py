import numpy as np

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4 * x) + np.cos(2 * x)

IG_max = 0 # начальное значение IG для сравнения
b = np.mean(y) # константное значение для расчета H(R) - начального impurity
# начальное impurity для всей выборки - мы считаем его как сумму
HR = np.sum((b - y) ** 2) 

for t in x: # перебираем различные значения порога
	# константные значения в листьях дерева
	b1, b2 = np.mean(y[x < t]), np.mean(y[x >= t])
	# impurity для 2 выборок
	HR1, HR2 = np.sum((b1 - y[x < t]) ** 2), np.sum((b2 - y[x >= t]) ** 2)
	IG = HR - len(y[x < t]) / len(y) * HR1 - len(y[x >= t]) / len(y) * HR2

	if IG > IG_max:
		IG_max = IG
		th = t

IG = IG_max

print(t, IG_max)