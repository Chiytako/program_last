import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import torch

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True
)

print(f'dataset size: {len(ds_train)}')

image, target = ds_train[0]
print(type(image))
print(target)

plt.imshow(image, cmap='gray_r', vmin=0, vmax=255)
plt.title(target)
plt.show()

image = transforms.functional.to_image(image)
image = transforms.functional.to_dtype(image, dtype=torch.float32, scale=True)
print(image.shape, image.dtype)
print(image.min(), image.max())
