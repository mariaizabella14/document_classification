
import glob
import pathlib

import numpy as np
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

train_path = 'dataset/train'
test_path = 'dataset/test'
pred_path = 'dataset/pred'

transformer = transforms.Compose([
    transforms.Resize((250,250)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transformer2 = transforms.Compose([
    transforms.Resize((250,250)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize([0.5], [0.5])
])

test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=64, shuffle=True
)

test_loader2 = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer2),
    batch_size=64, shuffle=True
)

root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()

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



    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)


        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.pool(output)

        output = output.view(-1, 50 * 14 * 14)
        output = self.fc1(output)
        output = self.fc2(output)


        return output


checkpoint = torch.load('best_checkpoint1.model')
model = ConvNet(num_classes=10)
#loading model
model.load_state_dict(checkpoint)
model.eval()


def prediction(img_path, transformer):
    image = Image.open(img_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    input = Variable(image_tensor)
    output = model(input)
    index = output.data.numpy().argmax()
    pred = classes[index]
    return pred

images_path=glob.glob(pred_path+'/*.jpg')

pred_dict = {}

for i in images_path:
    pred_dict[i[i.rfind('/')+1:]] = prediction(i,transformer)

print(pred_dict)



y_pred = []
y_true = []

# iterating over testing dataset
for inputs, labels in test_loader2:
    #feeding the model
    output = model(inputs)

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    #saving predictions
    y_pred.extend(output)
    #saving true classes for images
    labels = labels.data.cpu().numpy()
    y_true.extend(labels)


classes = ('ADVE', 'Email', 'Form', 'Letter', 'Memo',
           'News', 'Note', 'Report', 'Resume', 'Scientific')

# Building confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix , index=[i for i in classes],
                     columns=[i for i in classes])
plt.figure(figsize=(12, 12))
plt.show()
sn.heatmap(df_cm, annot=True)
plt.savefig('confusion_matrix.png')