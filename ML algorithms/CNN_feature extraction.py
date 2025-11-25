import torch
import torch.nn as nn
import torchvision.transforms.v2 as tfs
from PIL import Image
import matplotlib.pyplot as plt

file_path = r'C:\Users\eyvra\Desktop\DS\Projects\img.jpg'
img = Image.open(file_path)
# img.show()

# используем класс Compose, чтобы задать необходимые трансформации
# добавляем внешнюю ось batch_size
transforms = tfs.Compose([tfs.Grayscale(),
                          tfs.PILToTensor(),
                          tfs.ToDtype(torch.float32, scale=True),
                          tfs.Lambda(lambda x: x.unsqueeze_(dim=0))
                          ])

model = nn.Sequential(
                     nn.Conv2d(1, 4, kernel_size=10),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=5),
                     nn.Conv2d(4, 8, kernel_size=10),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=5),
                     )

t_in = transforms(img)

model.eval()
with torch.no_grad():
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 6))
    ax = ax.flatten()
    # пройдемся по слоям сети в цикле
    for i in range(len(model)):
        layer = model[i]
        t_in = layer(t_in)
        
        # выход второго слоя MaxPool2d
        if i == 5:
            for j, im in enumerate(torch.squeeze(t_in, 0)):
                # im - двумерный тензор, при подаче в ToPILImage 
                # двумерного тензора применяется цветовое кодирование LAB -
                # L (яркость), A (зелёный-пурпурный) и B (синий-жёлтый)
                img_out = tfs.ToPILImage()(im)
                ax[j].imshow(img_out)
                ax[j].axis('off') # убираем элементы осей
             
plt.tight_layout()
plt.show()