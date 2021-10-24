import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMGSIZE = 512 if torch.cuda.is_available() else 128

CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
CNN_NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)

CONTENT_LAYERS_DEFAULT = ['conv_4']
STYLE_LAYERS_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
