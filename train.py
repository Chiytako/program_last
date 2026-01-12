import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import v2 as transforms
import models

ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)

ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

batch_size = 64

dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

model = models.MyModel()

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_epochs = 20

train_loss_log = []
val_loss_log = []
train_acc_log = []
val_acc_log = []

for epoch in range(n_epochs):
    print(f'epoch {epoch+1}/{n_epochs}')
    
    train_loss = models.train(model, dataloader_train, loss_fn, optimizer)
    train_loss_log.append(train_loss)
    
    val_loss = models.test(model, dataloader_test, loss_fn)
    val_loss_log.append(val_loss)
    
    train_acc = models.test_accuracy(model, dataloader_train)
    train_acc_log.append(train_acc)

    val_acc = models.test_accuracy(model, dataloader_test)
    val_acc_log.append(val_acc)
    
    print(f'  train loss: {train_loss:.4f}, val loss: {val_loss:.4f}')
    print(f'  train acc:  {train_acc*100:.3f}%, val acc:  {val_acc*100:.3f}%')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, n_epochs+1), train_loss_log, label='train')
plt.plot(range(1, n_epochs+1), val_loss_log, label='validation')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.xticks(range(1, n_epochs+1, 2))
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, n_epochs+1), train_acc_log, label='train')
plt.plot(range(1, n_epochs+1), val_acc_log, label='validation')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.xticks(range(1, n_epochs+1, 2))
plt.grid()
plt.legend()

plt.show()
