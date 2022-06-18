import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries


#define how to get image
def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

#path to one single image
img = get_image('dataset/pred/11302743.jpg')
plt.imshow(img)
plt.show()

#define how to transform image
def get_input_transform():

    transf = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize([0.5], [0.5])
    ])

    return transf

#transforming single image in a batch of 1
def get_input_tensors(img):
    transf = get_input_transform()
    return transf(img).unsqueeze(0)


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

#loading model
checkpoint = torch.load('best_checkpoint1.model')
model = ConvNet(num_classes=10)
model.load_state_dict(checkpoint)


def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((250, 250))
    ])

    return transf

def get_preprocess_transform():

    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize([0.5], [0.5])
    ])

    return transf

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

test_pred = batch_predict([pill_transf(img)])
print(test_pred.squeeze().argmax())

#explanation for the image
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                         batch_predict,
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000)


#using a mask to output region of interest(area to sustain top prediction)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
img_boundry_sustain = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry_sustain)
plt.show()

#using a mask to see area against top prediction
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
img_boundry_against = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry_against)
plt.show()