import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os 

img_dir = './images'

transform = transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
img_name = 'neckarfront.jpg'
img_name = os.path.join(img_dir,
                                img_name)
image = Image.open(img_name)
image = transform(image)

model = models.vgg19(pretrained=True)
# print(model)

# Remove linear and pool layers (since we're not doing classification)
modules = list(model.children())[:-2]
model = nn.Sequential(*modules)
# print(model)


print(model(image.unsqueeze(0)).shape)
