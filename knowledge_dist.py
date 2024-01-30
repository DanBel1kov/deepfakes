"""Module for training a ConvNet model using knowledge distillation"""
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from tqdm import tqdm
from models.ConvNet import ConvNet
from models.ImageDatasetFullyRAM import ImageDatasetFullyRAM

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_test_dataloader(fp='data'):
    """Creates dataloader objects for train and test datasets"""
    transform2 = transforms.Compose(

        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ]
    )

    # trainset = datasets.ImageFolder(root=fp + "/train", transform=transform2)
    trainset = ImageDatasetFullyRAM(fp + "/train", transform=transform2)
    _trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = datasets.ImageFolder(root=fp + "/test", transform=transform2)
    _testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    return _trainloader, _testloader


def load_trained_resnet(fp='artifacts/resnet34_0.81_on20epoch_adam(0.0001), batch_size 16 Imagenet.pth'):
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
    return nn.functional.mse_loss(student_probs, teacher_probs)


trainloader, testloader = train_test_dataloader()
loss_fn = nn.BCEWithLogitsLoss()

teacher_model = load_trained_resnet()
student_model = ConvNet()

teacher_model = teacher_model.to(device)
student_model = student_model.to(device)

optimizer = optim.Adam(student_model.parameters(), lr=0.00003)

accs = []

for epoch in tqdm(range(1), leave=False):
    student_model.train()
    running_loss = 0
    for i, (images, labels) in enumerate(tqdm(trainloader)):
        images, labels = images.to(device), labels.to(device)
        labels = labels.float()

        with torch.no_grad():
            teacher_output = teacher_model(images)

        student_output = student_model(images)

        true_loss = loss_fn(student_output, labels.unsqueeze(1))

        distill_loss = distillation_loss(student_output, teacher_output)

        loss = 0.5 * true_loss + 5 * distill_loss
        running_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 25 == 24:
            # print('On step: ', i, 'current loss: ', round(running_loss / 25, 2))
            # print(f'On step {i} current loss: {(running_loss / 25):.2f}')
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 25))
            running_loss = 0

    class_correct = [0, 0]
    class_total = [0, 0]
    classes = ['Real', 'Fake']
    student_model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)

            y_pred = student_model(images)

            predicted = torch.squeeze(torch.round(y_pred))
            # print(predicted)
            # print(labels)
            c = predicted.cpu().detach() == labels

            # print(labels)
            for i, label in enumerate(labels):
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print(class_correct, class_total)
    acc = sum(class_correct) / sum(class_total)
    print(acc)
    accs.append(acc)

print(accs)
torch.save(student_model.state_dict(), 'artifacts/student_pro.pt')
