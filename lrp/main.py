# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# # Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.

#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
from torch import cuda
import torch
from torch import nn
from torchvision import transforms, models
import torchvision
from torchvision.models import vgg16, VGG16_Weights
from model_lrp import LRPModel
from visualize import plot_relevance_scores
import os



if __name__ == '__main__':

    # if cuda.is_available:
    #     device = torch.device("cuda")
    # else:
    device = torch.device("cpu")

    print(f"Using: {device}\n")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])

    # dataset_folder = os.path.join('lrp', 'images')
    dataset_folder = os.getcwd() + '/images'
    dataset = torchvision.datasets.ImageFolder(root=dataset_folder, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    model = models.efficientnet_b1(pretrained = True)

    model.classifier[1] = nn.Sequential(
        nn.Linear(1280, 1),
        nn.Sigmoid()
    )
    model_path = os.getcwd() + '/models/effnet_0.88.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    model.to(device)

    lrp_model = LRPModel(model=model)

    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        r = lrp_model.forward(x)
        print(f"Image {0} was read".format(i))
        plot_relevance_scores(x=x, r=r, name=str(i))
#%%

#%%
