'''First generation of simple CNN'''
import os

import torchvision
from torchvision.io import read_image

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim

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
            image = read_image(os.path.join(
                self.data_dir, 'fake', self.files[idx]))
        else:
            image = read_image(os.path.join(
                self.data_dir, 'real', self.files[idx]))

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
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=5,
                stride=3,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=5,
                stride=3,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2))
        self.drop_out = nn.Dropout(0.2)
        self.fc1 = nn.Linear(18496, 1000)
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, x):
        '''Feedforward the input through the CNN'''
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


NUM_EPOCHS = 50

print('Loading datasets. This may take some time... (check your RAM usage xd)')
image_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor()])
training_data = ImageDatasetFullyRAM(
    'data\\training', transform=image_transform)
train_dataloader = DataLoader(
    training_data, batch_size=64, shuffle=True, num_workers=0)
validation_data = ImageDatasetFullyRAM(
    'data\\validation', transform=image_transform)
validation_dataloader = DataLoader(
    validation_data, batch_size=100, shuffle=True, num_workers=0)
print('Datasets are loaded')


def objective(current_trial):
    '''Objective for Optuna search'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_step = len(train_dataloader)
    criterion = nn.BCEWithLogitsLoss()

    model = ConvNet()
    model = nn.DataParallel(model)
    model.to(device)

    optimizer_name = current_trial.suggest_categorical(
        "optimizer", ["Adam", "RMSprop", "SGD"])
    lr = current_trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    loss_list = []
    acc_list = []

    for epoch in range(NUM_EPOCHS):
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
                print(
                    f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{total_step}],'
                    f' Loss: {loss.item():.4f}, Accuracy: {(correct / total) * 100:.2f}%')
                loss_list.append(loss.item())
                acc_list.append(correct / total)

        # test
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
        current_trial.report(accuracy, epoch)
        print(f'Validated accuracy {accuracy}')
        if current_trial.should_prune():
            print('PRUNED')
            # dist.destroy_process_group()
            raise optuna.exceptions.TrialPruned()

    # dist.destroy_process_group()
    return accuracy


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=None)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    df = study.trials_dataframe()
    df.to_csv('models/optuna_results.csv', index=False)
    fig = optuna.visualization.plot_intermediate_values(study)
    fig.show()
