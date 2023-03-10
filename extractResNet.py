import torch
import torch.nn as nn
import torchvision.models as models
import os
from PIL import Image
from torchvision import transforms

"""
This file extracts ResNet features of images beforehand.
This process is not done during training because of computational limitations.
With minimal modficiation, it can be done jointly during training too
"""

# Image preprocessing with predefined tranforms
train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

val_transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

# load pretrained resnet
resnet = models.resnet152(pretrained=True)
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
# freeze layers
for param in resnet.parameters():
    param.requires_grad = False
resnet.cuda()

# train_images = os.listdir('../dataset/train')
# val_images = os.listdir('../dataset/val')
test_images = os.listdir('../dataset/test')

# we only need the features, disregard images
for file in test_images:
    image = Image.open('../dataset/test/' + file).convert('RGB')
    transformed = test_transform(image).cuda()
    out = resnet(torch.unsqueeze(transformed, dim=0))
    # make sure to use same name wit corresponding image
    torch.save(out, 'dataset/testfeatures/'+file.split('.')[0]+'.pt')
