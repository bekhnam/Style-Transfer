from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

import config

loader = transforms.Compose([
    transforms.Resize(config.IMGSIZE),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()

def content_image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(config.DEVICE, torch.float)

def style_image_loader(style_image_name, content_image_name):
    content_image = Image.open(content_image_name)
    image = Image.open(style_image_name).resize(content_image.size, resample=Image.BILINEAR)
    image = loader(image).unsqueeze(0)
    return image.to(config.DEVICE, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)