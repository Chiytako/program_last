import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import models

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = models.MyModel().to(device)

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])
)

image, target = ds_train[0]

image_tensor = image.unsqueeze(dim=0).to(device)

model.eval()
with torch.no_grad():
    logits = model(image_tensor)

logits = logits.cpu()
print(logits)

probs = logits.softmax(dim=1)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[0], cmap='gray_r', vmin=0, vmax=1)
plt.title(f'class: {target} ({datasets.FashionMNIST.classes[target]})')

plt.subplot(1, 2, 2)
plt.bar(range(len(probs[0])), probs[0])
plt.ylim(0, 1)
plt.title(f'predicted class: {probs[0].argmax()}')

plt.show()
