'''Combined model that uses EfficientNet, ResNet34 and GoogleNet'''
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
from tqdm import tqdm

torch.manual_seed(69)

TRAINING_FOLDER = 'data/training'
TEST_FOLDER = 'data/validation'

transform2 = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
trainset = datasets.ImageFolder(root=TRAINING_FOLDER, transform=transform2)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

testset = datasets.ImageFolder(root=TEST_FOLDER, transform=transform2)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def evaluate_model(model):
    '''Evaluate model accuracy using validation dataset'''
    class_correct = [0, 0]
    class_total = [0, 0]
    best_acc = 0.0
    acc = []
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)

            y_pred = model(images)

            predicted = torch.squeeze(torch.round(y_pred))

            c = predicted.cpu().detach() == labels

            for label, i in enumerate(labels):
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Total Accuracy is: ', sum(class_correct) / sum(class_total))
    accuracy = sum(class_correct) / sum(class_total)
    acc.append(sum(class_correct) / sum(class_total))
    if accuracy > best_acc and accuracy > 0.72:
        print("New Best Score!!! ")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(10, 1, 1)
    ax.plot(np.arange(len(acc)), acc)
    plt.show()

    return acc


def train_model(model, loss_fn, optimizer, scheduler=None, num_epochs=10, effnet=False):
    '''Train the model displaying process with tqdm'''
    best_model_wts = model.state_dict()

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        model.train()

        for i, batch in enumerate(tqdm(trainloader)):
            x_batch, y_batch = batch
            y_batch = y_batch.float()
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch.unsqueeze(1))

            if not effnet:
                loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            if i % 25 == 24:
                print(f'[{epoch + 1:d}, {i + 1:5d}] loss: {running_loss / 25:.3f}')
                running_loss = 0.0

    model.load_state_dict(best_model_wts)
    return model


class ResNet:
    def __init__(self):
        resnet = models.resnet34(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 1),
            nn.Sigmoid()
        )
        self.resnet = resnet.to(device)
        self.acc = []

    def train(self):
        '''Specify hyperparameters and train the model'''
        loss_fn2 = torch.nn.BCEWithLogitsLoss()
        optimizer3 = optim.Adam(self.resnet.parameters(), lr=0.0001)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer3,
                                               step_size=10,
                                               gamma=1)
        self.resnet, self.acc = train_model(self.resnet, loss_fn2,
                                            optimizer3, exp_lr_scheduler, num_epochs=10)

    def save(self, directory='models'):
        '''Save the trained model'''
        torch.save(self.resnet.state_dict(), f'{directory}/resnet_{str(self.acc[-1])}.pth')


class GNet:
    def __init__(self):
        gnet = models.googlenet(pretrained=True)
        gnet.fc = nn.Sequential(
            nn.Linear(1024, 1, bias=True),
            nn.Sigmoid()
        )
        self.gnet = gnet.to(device)
        self.acc = []

    def train(self):
        '''Specify hyperparameters and train the model'''
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.gnet.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)
        self.gnet, self.acc = train_model(self.gnet, loss_fn,
                                          optimizer, scheduler, num_epochs=15, effnet=False)

    def save(self, directory='models'):
        '''Save the trained model'''
        torch.save(self.gnet.state_dict(), f'{directory}/gnet_{str(self.acc[-1])}.pth')


if __name__ == '__main__':
    print(f"Using device: {device}")
    models = [ResNet(), GNet()]
    for cur_model in models:
        cur_model.train()
        cur_model.save()
