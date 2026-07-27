import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as tfs_v2
from tqdm import tqdm
import matplotlib.pyplot as plt

class AutoEncoderMNIST(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, 128),
			nn.ELU(inplace=True),
			nn.BatchNorm1d(128),
			nn.Linear(128, 64),
			nn.ELU(inplace=True),
			nn.BatchNorm1d(64)
			)

		# Слой для формирования среднего значения
		self.h_mean = nn.Linear(64, self.hidden_dim)
		# Слой для формирования логарифма дисперсии.
		self.h_log_var = nn.Linear(64, self.hidden_dim)

		self.decoder = nn.Sequential(
			nn.Linear(self.hidden_dim, 64),
			nn.ELU(inplace=True),
			nn.BatchNorm1d(64),
			nn.Linear(64, 128),
			nn.ELU(inplace=True),
			nn.BatchNorm1d(128),
			nn.Linear(128, output_dim),
			nn.Sigmoid()
			)

	def forward(self, x):
		enc = self.encoder(x)
		h_mean = self.h_mean(enc)
		h_log_var = self.h_log_var(enc)

		noise = torch.normal(mean=torch.zeros_like(h_mean),
							 std=torch.ones_like(h_log_var)
							 )
		h = noise * torch.exp(h_log_var / 2) + h_mean
		x = self.decoder(h)
		return x, h, h_mean, h_log_var


class VAELoss(nn.Module):
	def forward(self, x, y, h_mean, h_log_var):
		# x – тензор входных данных;
		# y – тензор выходных данных;
		# h_mean – мини-батч средних значений;
		# h_log_var - мини-батч логарифмов дисперсий
		img_loss = torch.sum(torch.square(x - y), dim=-1)
		kl_loss = -0.5 * torch.sum(1 + h_log_var - torch.square(h_mean) - \
								   torch.exp(h_log_var), dim=-1)
		return torch.mean(img_loss + kl_loss)


model = AutoEncoderMNIST(784, 784, 2)
transforms = tfs_v2.Compose([tfs_v2.ToImage(),
							 tfs_v2.ToDtype(dtype=torch.float32, scale=True),
							 tfs_v2.Lambda(lambda x: x.ravel())
							 ])
d_train = torchvision.datasets.MNIST(
	r'C:\Users\eyvra\Desktop\DS\Projects\VAE_MNIST\dataset',
	download=False,
	train=True,
	transform=transforms
	)
d_test = torchvision.datasets.MNIST(
	r'C:\Users\eyvra\Desktop\DS\Projects\VAE_MNIST\dataset',
	download=True,
	train=False,
	transform=transforms
	)
train_data = data.DataLoader(d_train, batch_size=128, shuffle=True)
# Для тестовой выборки обойдемся без даталоадера
test_data = transforms(d_test.data).view(len(d_test), -1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = VAELoss()
epochs = 10

# model.train()
# for _e in range(epochs):
# 	loss_mean = 0
# 	lm_count = 0

# 	train_tqdm = tqdm(train_data, leave=True)
# 	for x_train, y_train in train_tqdm:
# 		predict, _, h_mean, h_log_var = model(x_train)
# 		loss = loss_func(x_train, predict, h_mean, h_log_var)

# 		optimizer.zero_grad()
# 		loss.backward()
# 		optimizer.step()

# 		lm_count += 1
# 		loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
# 		train_tqdm.set_description(f'''Epoch [{_e + 1}/{epochs}],
# 									   loss_mean={loss_mean:.3f}''')

# st = model.state_dict()
# torch.save(st, 'model_vae.tar')

# Загрузка ранее сохраненных весов
weights = torch.load('model_vae.tar')
model.load_state_dict(weights)

model.eval()
with torch.no_grad():
	_, h, _, _ = model(test_data)

h = h.detach().numpy()

# Функции для различной визуализации
def space_visualize(h):
	# Визуализация скрытого пространстваы
	plt.scatter(h[:, 0], h[:, 1])
	plt.grid()
	plt.show()


def digits_visualize(h):
	# Возьмем диапазон [-3, 3] для оценки полученных изображений
	n = 5
	total = 2 * n + 1
	plt.figure(figsize=(total, total))

	num = 1
	for i in range(-n, n + 1):
		for j in range(-n, n + 1):
			ax = plt.subplot(total, total, num)
			num += 1
			h = torch.tensor([3 * i / n, 3 * j / n], dtype=torch.float32)
			predict = model.decoder(h.unsqueeze(0))
			predict = predict.detach().squeeze(0).view(28, 28)
			dec_img = predict.numpy()

			plt.imshow(dec_img, cmap='gray')
			# Современный вариант для ax.get_xaxis
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)

	plt.show()


def one_digit_visualize(h):
	# Возьмем изображения только цифры 1. Большинство остальных цифр получаются
	# хуже
	x_data = d_train.data[d_train.targets == 1]
	batch_size = x_data.size(0)
	x_data = transforms(x_data).view(batch_size, -1)
	enc = model.encoder(x_data)
	h_mean, h_log_var = model.h_mean(enc), model.h_log_var(enc)

	h_mean = torch.mean(h_mean, dim=0)
	h_std = torch.mean(torch.exp(h_log_var / 2), dim=0)

	n = 5
	total = 2 * n + 1
	plt.figure(figsize=(total, total))

	num = 1
	for i in range(-n, n + 1):
		for j in range(-n, n + 1):
			ax = plt.subplot(total, total, num)
			num += 1
			h = torch.tensor([3 * h_std[0] * i / n + h_mean[0],
							  3 * h_std[1] * j / n + h_mean[1]],
							  dtype=torch.float32)
			predict = model.decoder(h.unsqueeze(0))
			predict = predict.detach().squeeze(0).view(28, 28)
			dec_img = predict.numpy()

			plt.imshow(dec_img, cmap='gray')
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)

	plt.show()


one_digit_visualize(h)