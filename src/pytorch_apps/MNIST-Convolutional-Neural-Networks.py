import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.1307, std=0.3081)
])

mnist_data = datasets.MNIST(root='data/', download=True, transform=transformation, )

print(len(mnist_data))

train_data, valid_data = random_split(mnist_data, [50000, 10000])

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
val_loader = DataLoader(train_data, batch_size=256, shuffle=True)


class ResidualClassifier(nn.Module):
    def __init__(self):
        super(ResidualClassifier, self).__init__()

        self.r1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.s1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.r2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.s2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.fn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.r1(x)
        x2 = self.s1(x1)
        x3 = x1 + x2
        x4 = self.r2(x3)
        x5 = self.s2(x4)
        x6 = x4 + x5
        x7 = self.fn(x6)
        return x7, _


model = ResidualClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

total_step_train = len(train_loader)
total_step_validation = len(val_loader)
num_epochs = 3

training_losses = []
validation_losses = []

training_accuracy = []
validation_accuracy = []

for epoch in range(num_epochs):

    try:

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            targets = logits.argmax(dim=1)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_losses.append(loss.item())
            training_accuracy.append((targets == labels).sum().item() / images.shape[0])

        with torch.no_grad():

            model.eval()
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images)
                targets = logits.argmax(dim=1)

                loss = criterion(logits, labels)

                validation_losses.append(loss.item())
                validation_accuracy.append((targets == labels).sum().item() / targets.shape[0])

        training_loss = sum(training_losses[-total_step_train:]) / total_step_train
        validation_loss = sum(validation_losses[-total_step_validation:]) / total_step_validation

        at = sum(training_accuracy[-total_step_train:]) / total_step_train
        av = sum(validation_accuracy[-total_step_validation:]) / total_step_validation

        print(f"epoch: {epoch + 1} training_loss: {training_loss} validation_loss: {validation_loss} "
              f"training_accuracy: {at} "
              f"validation_accuracy: {av} "
              f"lr: {scheduler.get_last_lr()}"
              )

        scheduler.step()

    except KeyboardInterrupt:
        print('End.')
        break

training_losses, validation_losses = np.array(training_losses), np.array(validation_losses)

training_losses = training_losses.reshape(num_epochs, -1)
validation_losses = validation_losses.reshape(num_epochs, -1)

training_accuracy, validation_accuracy = np.array(training_accuracy), np.array(validation_accuracy)

training_accuracy = training_accuracy.reshape(num_epochs, -1)
validation_accuracy = validation_accuracy.reshape(num_epochs, -1)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

sns.lineplot(data=training_accuracy.mean(axis=1), label='training_accuracy', ax=axes[1, 0], color='blue')
sns.lineplot(data=validation_accuracy.mean(axis=1), label='validation_accuracy', ax=axes[1, 1], color='blue')

sns.lineplot(data=training_losses.mean(axis=1), label='training_losses', ax=axes[0, 0], color='red')
sns.lineplot(data=validation_losses.mean(axis=1), label='validation_losses', ax=axes[0, 1], color='red')

test_data = datasets.MNIST(root='data/', download=True, transform=transformation, train=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
print(len(test_data))

test_losses, test_accuracy = [], []

with torch.no_grad():
    model.eval()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        targets = logits.argmax(dim=1)

        loss = criterion(logits, labels)

        test_losses.append(loss.item())
        test_accuracy.append((targets == labels).sum().item())

test_loss = sum(test_losses) / len(test_data)
test_acc = sum(test_accuracy) / len(test_data)

print(f"test_loss: {test_loss} test_accuracy: {test_acc}")
