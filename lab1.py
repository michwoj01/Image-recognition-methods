import torch
import torchvision.transforms as T
import torchvision
from PIL import Image
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import ndimage

image = Image.open('./mro.png')
image_tensor = T.ToTensor()(image)
image_tensor = image_tensor.unsqueeze(0)

conv1x1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
weights = torch.tensor([0.3, 0.5, 0.1]).view(1, 3, 1, 1)
conv1x1.weight.data = weights
output = conv1x1(image_tensor)

output_image = output.squeeze().detach().numpy()
output_image = Image.fromarray((output_image * 255).astype('uint8'))
output_image.save('grayscale_result.png')

pooling = nn.MaxPool2d(kernel_size=4)
output = pooling(output)

output_image = output.squeeze().detach().numpy()
output_image = Image.fromarray((output_image * 255).astype('uint8'))
output_image.save('pooling_result.png')

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

gauss_blur = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
weights = torch.tensor(gaussian_kernel(5)).float().view(1, 1, 5, 5)
gauss_blur.weight.data = weights
output = gauss_blur(output)

output_image = output.squeeze().detach().numpy()
output_image = Image.fromarray((output_image * 255).astype('uint8'))
output_image.save('gaus_blur_result.png')

sobel_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
weights = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]).float().view(2, 1, 3, 3)
sobel_conv.weight.data = weights
output = sobel_conv(output)

x = output[:, 0, :, :]
y = output[:, 1, :, :]
euklides = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
euklides = euklides / euklides.max() * 255
arctan = torch.arctan2(y, x)

output_image = euklides.squeeze().detach().numpy()
output_image = Image.fromarray((output_image).astype('uint8'))
output_image.save('gradient_result.png')

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 1
                r = 1
                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]
                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0
            except IndexError as e:
                pass
    return Z

nms = non_max_suppression(euklides.squeeze().detach().numpy(), arctan.squeeze().detach().numpy())
output[:, 0, :, :] = torch.tensor(nms)

output = output[:, 0:1, :, :]

output_image = output.squeeze().detach().numpy()
output_image = Image.fromarray((output_image).astype('uint8'))
output_image.save('non_max_result.png')

threshold = nn.Threshold(50, 0)
output = threshold(output)
output = torch.where(output > 0, torch.ones_like(output), torch.zeros_like(output))

output_image = output.squeeze().detach().numpy()
output_image = Image.fromarray((output_image * 255).astype('uint8'))
output_image.save('threshold_result.png')

upscale = nn.Upsample(scale_factor=4, mode='nearest')
output = upscale(output)

padding = (1, 1, 1, 1)  
output = nn.functional.pad(output, padding)

output_image = output.squeeze().detach().numpy()
output_image = Image.fromarray((output_image * 255).astype('uint8'))
output_image.save('upscale_result.png')

image_tensor[:,1,:,:] = image_tensor[:,1,:,:] + output

output_image = (image_tensor * 255).clamp(0, 255).byte()  # Ensure values are between 0 and 255 and use byte data type

# Convert the NumPy array to a PIL Image
output_image = Image.fromarray(output_image.squeeze().permute(1, 2, 0).numpy())

# Save the image
output_image.save('final_result.png')

