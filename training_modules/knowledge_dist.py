"""Module for training a ConvNet model using knowledge distillation"""
import optuna
import torch
from optuna.trial import TrialState
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
from tqdm import tqdm
from models.ConvNet import ConvNet

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(69)


def train_test_dataloader(fp='data'):
    """Creates dataloader objects for train and test datasets"""
    transform2 = transforms.Compose(

        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ]
    )

    trainset = datasets.ImageFolder(root=fp + "/train", transform=transform2)
    # trainset = ImageDatasetFullyRAM(fp + "/train", transform=transform2)
    _trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.ImageFolder(root=fp + "/test", transform=transform2)
    _testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    return _trainloader, _testloader


def load_trained_resnet(fp='artifacts/resnet_09964_sd.pt'):
    """Loads a pretrained resnet34 for knowledge distillation"""
    resnet = models.resnet34(pretrained=True)
    resnet.fc = nn.Sequential(
        nn.Linear(resnet.fc.in_features, 1),
        nn.Sigmoid()
    )

    resnet.load_state_dict(torch.load(fp, map_location=device))

    resnet = resnet.to(device)
    return resnet


def distillation_loss(_student_output, _teacher_output, _t=2.0):
    """Loss function for knowledge distillation"""
    teacher_probs = torch.sigmoid(_teacher_output / _t)
    student_probs = torch.sigmoid(_student_output / _t)
    return nn.functional.mse_loss(student_probs, teacher_probs) * _t ** 2


def validate_model(_model, _testloader):
    """Uses a dataloader to validate the model. Outputs model's accuracy on a given dataset"""
    _model.eval()
    class_correct = [0, 0]
    class_total = [0, 0]
    # classes = ['Real', 'Fake']
    _model.eval()
    with torch.no_grad():
        for data in tqdm(_testloader, leave=False, position=0):
            images, labels = data
            images = images.to(device)

            y_pred = _model(images)
            predicted = torch.squeeze(torch.round(y_pred))
            c = predicted.cpu().detach() == labels

            for i, label in enumerate(labels):
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print(class_correct, class_total)
    _acc = sum(class_correct) / sum(class_total)
    return _acc


def know_dist_epoch(_model, _epoch_n, optimizer, teacher_model, lr_scheduler=None, alpha=0.1):
    """Run one epoch of training on a given model inplace
     _epoch_n is used to print the epoch number
     alpha is a coefficient for knowledge distillation"""
    _model.train()
    running_loss = 0

    with tqdm(trainloader, leave=False, position=0) as t:
        for i, (images, labels) in enumerate(t):
            images, labels = images.to(device), labels.to(device)
            labels = labels.float()

            with torch.no_grad():
                teacher_output = teacher_model(images)

            student_output = _model(images)

            true_loss = loss_fn(student_output, labels.unsqueeze(1))

            distill_loss = distillation_loss(student_output, teacher_output)

            loss = alpha * true_loss + (1 - alpha) * distill_loss
            running_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            batch_to_print = 100
            if i % batch_to_print == batch_to_print - 1:
                t.set_description(f'Epoch: {_epoch_n}, loss: {running_loss / batch_to_print}')
                running_loss = 0


trainloader, testloader = train_test_dataloader()
loss_fn = nn.BCEWithLogitsLoss()

teacher_model = load_trained_resnet()
student_model = ConvNet()

teacher_model = teacher_model.to(device)
student_model = student_model.to(device)

optimizer = optim.Adam(student_model.parameters(), lr=0.000947829)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=580, gamma=0.10962742, verbose=False)

accs = []

for epoch in tqdm(range(10), leave=False):
    know_dist_epoch(student_model, epoch, optimizer, teacher_model, exp_lr_scheduler, alpha=0.0515167)
    student_model.eval()
    acc = validate_model(student_model, testloader)
    accs.append(acc)

print(accs)
torch.save(student_model.state_dict(), f'artifacts/student_optuna_{accs[-1]}.pt')
