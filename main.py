import torch
import os

import torchvision
from torch import nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.optim as optim

import optuna
from optuna.trial import TrialState


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = os.listdir(os.path.join(self.data_dir, 'fake'))
        self.files.extend(os.listdir(os.path.join(self.data_dir, 'real')))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = 0 if 'real' in self.files[idx] else 1  # 0=REAL, 1=FAKE
        if label == 1:
            image = read_image(os.path.join(self.data_dir, 'fake', self.files[idx]))
        else:
            image = read_image(os.path.join(self.data_dir, 'real', self.files[idx]))

        if self.transform:
            image = self.transform(image)
        return image, label


class ImageDatasetFullyRAM(Dataset):  # loads the WHOLE dataset into RAM
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = os.listdir(os.path.join(self.data_dir, 'fake'))
        self.files.extend(os.listdir(os.path.join(self.data_dir, 'real')))
        self.items = []
        for file in self.files:
            label = 0 if 'real' in file else 1  # 0=REAL, 1=FAKE
            if label == 1:
                image = read_image(os.path.join(self.data_dir, 'fake', file))
            else:
                image = read_image(os.path.join(self.data_dir, 'real', file))

            if transform:
                image = transform(image)

            self.items.append((image, label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.items[idx]


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=3, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=3, padding=2),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout(0.2)
        self.fc1 = nn.Linear(18496, 1000)
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


num_epochs = 50

print('Loading datasets. This may take some time... (check your RAM usage xd)')
transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor()])
training_data = ImageDatasetFullyRAM('data\\training', transform=transform)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=0)
validation_data = ImageDatasetFullyRAM('data\\validation', transform=transform)
validation_dataloader = DataLoader(validation_data, batch_size=100, shuffle=True, num_workers=0)
print('Datasets are loaded')


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_step = len(train_dataloader)
    criterion = nn.BCEWithLogitsLoss()

    model = ConvNet()
    model = nn.DataParallel(model)
    model.to(device)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    loss_list = []
    acc_list = []

    for epoch in range(num_epochs):
        # training
        for i, (images, labels) in enumerate(train_dataloader):
            if device.type == 'cuda':
                images = images.cuda()
                labels = labels.cuda()
            # feed forward
            outputs = model(images)
            labels = labels.unsqueeze(1).float()
            loss = criterion(outputs, labels)

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accuracy
            total = labels.size(0)
            predicted = (outputs.data > 0.5).float()
            correct = (predicted == labels).sum().item()

            if i % 16 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
                loss_list.append(loss.item())
                acc_list.append(correct / total)

        # validation
        correct_count = 0
        for i, (images, labels) in enumerate(validation_dataloader):
            if device.type == 'cuda':
                images = images.cuda()
                labels = labels.cuda()
            # feed forward
            outputs = model(images)
            labels = labels.unsqueeze(1).float()

            predicted = (outputs.data > 0.5).float()
            correct_count += (predicted == labels).sum().item()

        accuracy = correct_count / len(validation_data)
        trial.report(accuracy, epoch)
        print(f'Validated accuracy {accuracy}')
        if trial.should_prune():
            print('PRUNED')
            # dist.destroy_process_group()
            raise optuna.exceptions.TrialPruned()

    # dist.destroy_process_group()
    return accuracy


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df = study.trials_dataframe()
    df.to_csv('models/optuna_results.csv', index=False)
    fig = optuna.visualization.plot_intermediate_values(study)
    fig.show()
