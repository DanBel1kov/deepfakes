from PIL import Image
from PIL import Image, ImageChops, ImageEnhance
import torch
import numpy as np
from torchvision import transforms, datasets
import os
from tqdm import tqdm_notebook
import torchvision
from torch import nn
from torch.utils.data import Dataset, ConcatDataset,  SubsetRandomSampler
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, models
from torch.optim import lr_scheduler

import torch.nn as nn
import torch.nn.functional as F
import os
import cv2

def convert_ela_transform(x, quality):

  resaved = '_resaved.jpeg'
  im = x.convert('RGB')
  im.save(resaved, 'JPEG', quality = quality )

  compressed_image = Image.open(resaved)
  os.remove(resaved)

  ela_im = ImageChops.difference(im, compressed_image)

  extrema = ela_im.getextrema()

  max_dif = max([ex[1] for ex in extrema])
  if max_dif == 0:
    max_diff = 1

  scale = 255.0 / max_dif

  ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

  return ela_im

def convert_to_ela(path, quality, fake, path_ = False):
  if not path_:
    resaved = '_resaved.jpeg'
    im = path.convert('RGB')
    path.save(resaved, 'JPEG', quality = quality)
    
  else:
    resaved = path.split('.')[0] + '_resaved.jpeg'
    ela_image_path = path.split('.')[0] + '_ela.png'
    im = Image.open(path).convert('RGB')
    im.save(resaved, 'JPEG', quality = quality )

  compressed_image = Image.open(resaved)
  os.remove(resaved)

  ela_im = ImageChops.difference(im, compressed_image)

  extrema = ela_im.getextrema()

  max_dif = max([ex[1] for ex in extrema])
  if max_dif == 0:
    max_diff = 1

  scale = 255.0 / max_dif

  ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

  plt.subplot(1, 2, 1)
  plt.imshow(im)
  plt.title("Fake Image" if fake else "Image")
  plt.axis('off')

  plt.subplot(1, 2, 2)
  plt.imshow(ela_im)
  plt.title("Fake Ela Image" if fake else "Ela Image")
  plt.axis('off')

  plt.show()


transform2  = transforms.Compose(                      #for ela training 

    [
     transforms.Resize((256, 256)),
     transforms.Lambda(lambda x: convert_ela_transform(x, quality=90)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

def plot_grad_cam(path):
      image = Image.open(path)
      transformed_image = transform(image)
      transformed_image = transformed_image.to(device).unsqueeze(0)

      pred = net(transformed_image).round()

      captured_gradients = net.get_activations_gradient()

      pooled_gradients = torch.mean(captured_gradients, dim = [0, 2, 3])
      activations = net.get_activations(transformed_image.to(device)).detach()

      for i in range(128):
        activations[:, i, :, :] *= pooled_gradients[i]

      activations 
      heatmap = torch.mean(activations, dim=1).squeeze()
      heatmap = np.maximum(heatmap.cpu(), 0)
      heatmap /= torch.max(heatmap)
      channel_to_visualize = heatmap[ :, :]

      # plt.matshow(channel_to_visualize)   plot heatmap
      # plt.show()

      img = cv2.imread(path)
      heatmap = channel_to_visualize.numpy()
      heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) 
      heatmap = np.uint8(255 * heatmap)

      # Resize the heatmap to match the image dimensions # 
      heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

      heat_pic = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

      superimposed_img = cv2.addWeighted(img, 1, heat_pic, 0.4, 0)

      plt.subplot(1, 2, 1)
      plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      plt.title('Original Image')
      plt.axis('off')

      plt.subplot(1, 2, 2)
      plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
      plt.title('Places of interests')
      plt.axis('off')

      plt.show()

plot_grad_cam(fake_path)
plot_grad_cam(real_path)
