"""Module for training a ConvNet model using knowledge distillation"""
import optuna
import torch
from optuna.trial import TrialState
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
from torchvision.models import ResNet34_Weights
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
    _trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    testset = datasets.ImageFolder(root=fp + "/test", transform=transform2)
    _testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    return _trainloader, _testloader


def load_trained_resnet(fp='artifacts/resnet_09964_sd.pt'):
    """Loads a pretrained resnet34 for knowledge distillation"""
    resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
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


def objective(current_trial):
    '''Objective for Optuna search'''

    # optimizer_name = current_trial.suggest_categorical(
    #     "optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer_name = "Adam"
    lr = current_trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    alpha = current_trial.suggest_float("alpha", 0.05, 0.5, log=True)
    gamma = current_trial.suggest_float("gamma", 1e-2, 1, log=True)
    step_size = current_trial.suggest_float("step_size", 300, 3000, log=True)

    student_model = ConvNet()
    # student_model = nn.DataParallel(student_model)
    teacher_model = load_trained_resnet()

    optimizer = getattr(optim, optimizer_name)(student_model.parameters(), lr=lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=False)

    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    accs = []

    for epoch in tqdm(range(7), leave=False):
        know_dist_epoch(student_model, epoch, optimizer, teacher_model, lr_scheduler=exp_lr_scheduler, alpha=alpha)
        student_model.eval()
        acc = validate_model(student_model, testloader)
        accs.append(acc)

        current_trial.report(acc, epoch)
        print(f'Validated accuracy {acc}')
        if current_trial.should_prune() or acc <= 0.5:
            print('PRUNED')
            raise optuna.exceptions.TrialPruned()

    return accs[-1]


trainloader, testloader = train_test_dataloader()
loss_fn = nn.BCEWithLogitsLoss()

# torch.save(student_model.state_dict(), f'artifacts/student_pro_{accs[-1]}.pt')

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=None)

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
    df.to_csv('artifacts/optuna_results_broad.csv', index=False)
    fig = optuna.visualization.plot_intermediate_values(study)
    fig.show()
