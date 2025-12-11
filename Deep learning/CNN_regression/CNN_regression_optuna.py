import os
import json
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import optuna

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

# определяем датасеты, используем валидационную выборку для подбора 
# гиперпараметров, разбиваем обучающую выборку на train и val
# через random_split (70/30)
d_train, d_val = data.random_split(SunDataset('dataset', transform=transforms),
                                   [0.7, 0.3])
d_test = SunDataset('dataset', train=False, transform=transforms)

# т.к. предсказываем координаты из 2 чисел, на выходе полносвязного слоя
# должно быть 2 нейрона
# nn.Flatten() вытягивает тензор, начиная с dim=1 по умолчанию (сохраняя
# ось batch_size)
# т.к. мы используем ранее найденные веса weights, то количество слоев НС
# должно им соответствовать (т.е. их не меняем)
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

loss_func = nn.MSELoss() # функция потерь

def objective(trial):
    """Функция для нахождения оптимальных гиперпараметров НС через optuna.
    Первоначально назначаются веса, найденные ранее без optuna"""
    # batch_size для train_data
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    # lr и weight_decay для оптимизатора
    lr = trial.suggest_float('lr', 1e-5, 0.1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 0.1, log=True)
 
    # определяем итераторы для извлечения выборок
    train_data = data.DataLoader(d_train, batch_size=batch_size, shuffle=True)
    val_data = data.DataLoader(d_val, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = 5 # количество эпох

    # загружаем ранее полученные веса
    # weights_only=True означает, что загружаются примитивные типы данных
    weights = torch.load('model_initial.tar', weights_only=True)
    model.load_state_dict(weights)

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
            loss_mean = 1 / lm_count * loss.item() \
                        + (1 - 1 / lm_count) * loss_mean
            # выводим номер эпохи и значение функции потерь
            train_tqdm.set_description(
                       f'Epoch [{i + 1} / {epochs}], loss_mean={loss_mean:.3f}'
                                      )

    # валидация модели
    model.eval()
    with torch.no_grad():
        Q_val = 0
        val_count = 0
        for x_val, y_val in val_data:
            y_pred_val = model(x_val)
            loss_val = loss_func(y_val, y_pred_val)

            val_count += 1
            Q_val = 1 / val_count * loss_val.item() \
                    + (1 - 1 / val_count) * Q_val

    return Q_val


study = optuna.create_study(direction='minimize', study_name='CNN')
study.optimize(objective, n_trials=5)

# выведем результаты
best_value = study.best_value
best_params = study.best_params
print(f'Best value: {best_value}')
print(f'Best params: {best_params}')