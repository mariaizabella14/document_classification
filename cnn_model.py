
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib


#checking for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transformer = transforms.Compose([
    transforms.Resize((250,250)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize([0.5], [0.5])
])

#Dataloader
#Path for training and testing directory
train_path = 'dataset/train'
test_path = 'dataset/test'

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=32, shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=64, shuffle=True
)

root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])   #listing subdirectories
print(classes)


dataset = ImageFolder(train_path, transform=transformer)
img, label = dataset[0]
print(img.shape, label)


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1
        # 𝑊 = input volume
        # 𝑃 = padding
        # 𝐹 = filtersize
        # 𝑆 = stride

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=20)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=4)

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=7, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_features=50)
        self.pool = nn.MaxPool2d(kernel_size=4)

        self.fc1 = nn.Linear(in_features=14 * 14 * 50, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

        # Feed forwad function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.pool(output)
        #print(output.shape)
        output = output.view(-1, 50 * 14 * 14)
        output = self.fc1(output)
        output = self.fc2(output)

        return output




model = ConvNet(num_classes=10).to(device)
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.1)
#optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

num_epochs = 20
train_count = len(glob.glob(train_path+'/**/*.jpg'))
test_count = len(glob.glob(test_path+'/**/*.jpg'))
print(train_count, test_count)

# Model training and saving best model

best_accuracy = 0.0
val_acc = []
train_acc = []

for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on testing dataset
    model.eval()
    test_accuracy = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count
    train_acc.append(train_accuracy)
    val_acc.append(test_accuracy)

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint1.model')
        best_accuracy = test_accuracy

plt.figure(figsize=(10,5))
plt.title("Training and Validation Accuracy")
plt.plot(val_acc,label="val")
plt.plot(train_acc,label="train")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('graph.png')
plt.show()
