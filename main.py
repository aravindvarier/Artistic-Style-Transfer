import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import copy
from collections import OrderedDict

img_dir = './images'
image_save_folder = './image_saves'
if not os.path.isdir(image_save_folder):
    os.mkdir(image_save_folder)

torch.manual_seed(42)

class My_VGG19(nn.Module):
    def __init__(self):
        super(My_VGG19, self).__init__()
        self.model = models.vgg19(pretrained=True)
        # print(model)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(self.model.children())[:-2]
        self.model = nn.Sequential(*modules)
        modules = list(self.model.children())[0]
        self.model = nn.Sequential(*modules)
        all_modules = list(self.model.children())
        self.model_parts = nn.ModuleList([nn.Sequential(*all_modules[:1]),
                            nn.Sequential(*all_modules[1:6]),
                            nn.Sequential(*all_modules[6:11]),
                            nn.Sequential(*all_modules[11:20]),
                            nn.Sequential(*all_modules[20:29])])

    def forward(self, x, num_parts_content, num_parts_style):
        style_outputs = []
        y = x
        for i in range(num_parts_content):
            y = self.model_parts[i](y)
        
        z = x

        for i in range(num_parts_style):
            z = self.model_parts[i](z)
            style_outputs.append(z)

        return y, style_outputs

        
        

        

transform = transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
content_img_name = 'neckarfront.jpg'
content_img_name = os.path.join(img_dir, content_img_name)
content_image = Image.open(content_img_name)
content_image = transform(content_image)
content_image = content_image.to('cuda')

style_img_name = 'persistenceofmemory.jpg'
style_img_name = os.path.join(img_dir, style_img_name)
style_image = Image.open(style_img_name)
style_image = transform(style_image)
style_image = style_image.to('cuda')


model = My_VGG19()
print("===================== MODEL ARCHITECTURE =====================")
print(model)
input("Press Return to continue or Ctrl-C to exit")
model.to('cuda')
x = torch.randn(1,3,224,224,device='cuda',requires_grad=True)

max_epochs = 10000
print_freq = 10
save_freq = 500
acceptable_diff = 50

optimizer = torch.optim.Adam([x], lr=0.001)
inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )



epoch = 1
x_prev = copy.deepcopy(x)

content_parts = 4
style_parts = 4
sizes = [] #Used to store the size of the feature maps of the intermediate outputs at each stage

while True:
    optimizer.zero_grad()
    content_image_output, _ = model(content_image.unsqueeze(0), content_parts, 0)    
    _, style_image_output = model(style_image.unsqueeze(0), 0, style_parts)
    for i in range(len(style_image_output)):
        sizes.append(style_image_output[i].shape[-1])
        style_image_output[i] = torch.flatten(style_image_output[i].squeeze(0), 1, 2)
        style_image_output[i] = torch.matmul(style_image_output[i], style_image_output[i].T)  #Gram matrix construction

    target_content_output, target_style_output = model(x, content_parts, style_parts)
    for i in range(len(target_style_output)):
        target_style_output[i] = torch.flatten(target_style_output[i].squeeze(0), 1, 2)
        target_style_output[i] = torch.matmul(target_style_output[i], target_style_output[i].T)  #Gram matrix construction

    content_loss = 0.5 * torch.sum((content_image_output - target_content_output)**2)

    #STYLE LOSS CALCULATION
    style_loss = 0
    for i in range(len(target_style_output)):
        style_loss += (1/(sizes[i]**2 * target_style_output[i].shape[-1]**2))*torch.sum((target_style_output[i] - style_image_output[i])**2)


    if epoch % print_freq == 0:
        print("Epoch: ", epoch, " Loss: ", style_loss.item())
    style_loss.backward()
    optimizer.step()

    if epoch % save_freq == 0:
        with torch.no_grad():
            diff = torch.sum((x_prev - x)**2).item()
            print("Difference from previous stored image: ", diff)
            if diff < acceptable_diff:
                print("The difference between images is now too low. Exiting.")
                break
            inverted = inv_normalize(x[0])
            save_image(inverted, os.path.join(image_save_folder, str(epoch)+".jpg"))
            x_prev = copy.deepcopy(x)

    epoch += 1



