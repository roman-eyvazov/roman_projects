import os
import json
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class SunDataset(data.Dataset):
    """Класс для формирования датасета с изображениями солнца"""
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, 'train' if train else 'test')
        self.transform = transform

        # открываем файл 'format.json' с названиями изображений
        # и таргетами (координатами солнца)
        with open(os.path.join(self.path, 'format.json'), 'r') as f:
            self.format = json.load(f)

        self.length = len(self.format) # размер выборки
        self.files = tuple(self.format.keys()) # кортеж с названиями изображений
        self.targets = tuple(self.format.values()) # кортеж с таргетами

    def __getitem__(self, item):
        path_file = os.path.join(self.path, self.files[item])
        # загрузим изображения, конвертация в RGB выполняется для случаев, 
        # когда исходные изображения имеют другое представление
        img = Image.open(path_file).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.targets[item], dtype=torch.float32)

    def __len__(self):
        return self.length


# на вход НС будут поступать изображения размерами 256х256
transforms = tfs.Compose([tfs.ToImage(),
                          tfs.ToDtype(torch.float32, scale=True)])

# определяем датасеты и итераторы для обучающей и тестовой выборок
d_train = SunDataset('dataset', transform=transforms)
d_test = SunDataset('dataset', train=False, transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)
# для test_data сделаем один батч на всю выборку
test_data = data.DataLoader(d_test, batch_size=len(d_test), shuffle=False)

# т.к. предсказываем координаты из 2 чисел, на выходе полносвязного слоя
# должно быть 2 нейрона
# nn.Flatten() вытягивает тензор, начиная с dim=1 по умолчанию (сохраняя
# ось batch_size)
model = nn.Sequential(nn.Conv2d(3, 32, 3, padding='same'),
                      nn.ReLU(),
                      nn.MaxPool2d(2),
                      nn.Conv2d(32, 8, 3, padding='same'),
                      nn.ReLU(),
                      nn.MaxPool2d(2),
                      nn.Conv2d(8, 4, 3, padding='same'),
                      nn.ReLU(),
                      nn.MaxPool2d(2),
                      nn.Flatten(),
                      nn.Linear(4096, 128),
                      nn.ReLU(),
                      nn.Linear(128, 2)
                      )

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
loss_func = nn.MSELoss() # функция потерь
epochs = 5 # количество эпох

# переводим модель в режим обучения
model.train()
for i in range(epochs):
    loss_mean = 0 # начальное значение скользящего среднего функции потерь
    lm_count = 0 # счетчик итераций градиентного спуска (пересчета loss)

    # leave=True оставляет предыдущий progressbar на экране
    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_func(y_train, predict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        # выводим номер эпохи и значение функции потерь
        train_tqdm.set_description(
                       f'Epoch [{i + 1} / {epochs}], loss_mean={loss_mean:.3f}'
                       )

# сохраним веса модели
st = model.state_dict()
torch.save(st, 'model_initial.tar')

# переводим модель в режим тестирования
model.eval()
with torch.no_grad():
    # берем все образы из test_data за раз
    x_test, y_test = next(iter(test_data))
    y_pred = model(x_test)
    Q = loss_func(y_test, y_pred).item() # эмпирический риск

print(f'MSELoss = {Q}')