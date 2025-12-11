import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import models
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim

# загрузим изображения, конвертация в RGB выполняется для случаев,
# когда исходные изображения имеют другое представление
img_original = Image.open('img_original.jpg').convert('RGB')
img_style = Image.open('img_style.jpg').convert('RGB')

# определим трансформации для входных изображений с учетом приведения к одному
# размеру; ToImage() преобразует изображение PIL к типу
# torchvision.tv_tensors._image.Image
transforms = tfs.Compose([tfs.ToImage(), 
						  tfs.Resize([400, 400]),
						  tfs.ToDtype(torch.float32, scale=True)])

# преобразуем изображения в тензоры с добавлением оси batch_size
img_original = transforms(img_original).unsqueeze_(dim=0)
img_style = transforms(img_style).unsqueeze_(dim=0)

# создадим тензор для формируемого изображения как копию исходного
# также необходимо включить для него градиенты
img_create = img_original.clone()
img_create.requires_grad_(True)

# чтобы можно было получать данные с разных сверточных слоев, создадим класс
class ModelStyle(nn.Module):
	"""Сверточная НС для стилизации изображений на основе сети VGG19"""
	def __init__(self):
		super().__init__()
		# используем обученную модель VGG19
		_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT, 
							  progress=True)
		# model.features - это только сверточные слои, полносвязные не нужны
		self.mf = _model.features
		# т.к. модель self.mf не будет обучаться, отключаем градиенты
		self.mf.requires_grad_(False)
		# также отключаем градиенты для всей модели
		self.requires_grad_(False)
		# переводим модель в режим тестирования
		self.mf.eval()
		self.idx_out = (0, 5, 10, 19, 28, 34) # индексы сверточных слоев VGG19
		# индекс последнего слоя для расчета потерь по контенту
		self.num_style_layers = len(self.idx_out) - 1

	def forward(self, x):
		# добавляем в список outputs тензоры с выходов интересующих нас
		# сверточных слоев
		outputs = []
		for idx, layer in enumerate(self.mf):
			x = layer(x)
			if idx in self.idx_out:
				outputs.append(x.squeeze_(dim=0))

		return outputs


model = ModelStyle()

# списки с тензорами с выходов интересующих сверточных слоев
outputs_img_original = model(img_original)
outputs_img_style = model(img_style)

def get_content_loss(base_content, target):
	"""Функция для расчета потерь по контенту; base_content - выходной тензор
	для формируемого изображения; target - выходной тензор для
	исходного изображения"""
	return torch.mean((base_content - target) ** 2)


def gram_matrix(x):
	"""Функция для расчета матриц Грама для тензоров с выходов сверточных слоев;
	на вход поступает тензор (channels, H, W)"""
	channels = x.size(dim=0)
	# делаем тензор двумерным с размерами (channels, (H * W))
	g = x.view(channels, -1)
	# вычисление матрицы Грама
	gram = (g @ g.mT) * g.size(dim=1)
	return gram


def get_style_loss(base_style, gram_target):
	"""Функция для расчета потерь по стилю; base_style - список тензоров со всех
	сверточных слоев формируемого изображения; gram_target - набор матриц Грама
  	для стилевого изображения"""
	# веса для различных слоев при расчете потерь
	style_weights = [1.0, 0.8, 0.5, 0.3, 0.1]
	_loss = 0 # начальное значение потерь
	i = 0 # счетчик итераций
	for base, target in zip(base_style, gram_target):
		gram_style = gram_matrix(base)
		_loss += style_weights[i] * torch.mean((gram_style - target) ** 2)
		i += 1

	return _loss


# список из матриц Грама для стилевого изображения
gram_matrix_style = [gram_matrix(x) for x
					 in outputs_img_style[:model.num_style_layers]]

# веса для потерь по контенту и стилю
content_weight = 1
style_weight = 1000
# сохраняем лучшие результаты стилизации с помощью best_loss и best_img
best_loss = -1
best_img = img_create.clone()
epochs = 100 # количество эпох
# оптимизируемые параметры – это пиксели формируемого изображения
optimizer = optim.Adam(params=[img_create], lr=0.01)

for i in range(epochs):
	outputs_img_create = model(img_create)
 
	# для расчета потерь по контенту берем тензоры с выходов последних слоев
	loss_content = get_content_loss(outputs_img_create[-1],
	                                outputs_img_original[-1])
	loss_style = get_style_loss(outputs_img_create, gram_matrix_style)
	# общие потери
	loss = content_weight * loss_content + style_weight * loss_style

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	# тензор img_create ограничивается диапазоном [0; 1] - то, что меньше 0,
	# равно 0; то, что больше 1, равно 1
	img_create.data.clamp_(0, 1)
 
	if loss < best_loss or best_loss < 0:
		best_loss = loss
		best_img = img_create.clone()

	print(f'Iteration: {i}, loss: {loss.item():.2f}')
 
# размерность best_img - (1, 3, 600, 600)
# отключаем градиенты и удаляем внешнюю ось
x = best_img.detach().squeeze_()
low, hi = torch.amin(x), torch.amax(x)
x = (x - low) / (hi - low) * 255.0
# нужно поменять оси, чтобы получить (H, W, channels); также
# конвертируем в массив numpy
x = x.permute(1, 2, 0).numpy()
x = np.clip(x, 0, 255).astype('uint8')

# сохраним изображение и выведем на экран
image = Image.fromarray(x, 'RGB')
image.save("result.jpg")

plt.imshow(x)
plt.show()