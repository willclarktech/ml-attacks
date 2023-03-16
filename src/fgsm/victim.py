from __future__ import print_function

import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(data_dir: str) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        data_dir,
        train=False,
        download=False,
        transform=transform,
    )
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset)
    return train_loader, test_loader


def create_model() -> nn.Sequential:
    # A very simple model which can be trained on a CPU for >95% accuracy on MNIST
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
        nn.LogSoftmax(dim=-1),
    )


def load_or_create_model(model_path) -> nn.Sequential:
    try:
        model = torch.load(model_path)
        print("Loaded pretrained model.")
        return model
    except:
        print("Pretrained model not found. Creating new model.")
        return create_model()


def train_epoch(
    model: nn.Sequential,
    loader: DataLoader,
    optimizer: optim.Optimizer,
) -> None:
    model.train()

    for inp, target in loader:
        optimizer.zero_grad()
        output = model(inp)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    print(f"Train set latest loss: {loss.sum().item()}")


def test(model: nn.Sequential, loader: DataLoader) -> None:
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inp, target in loader:
            output = model(inp)
            test_loss += float(F.nll_loss(output, target, reduction="sum").item())
            prediction = output.argmax(dim=-1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    print(
        f"Test set average loss: {test_loss/len(loader.dataset)}; accuracy: {correct}/{len(loader.dataset)}"  # type: ignore
    )


def train_model(
    model: nn.Sequential,
    n_epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
):
    optimizer = optim.Adam(model.parameters())
    for i in range(n_epochs):
        print(f"Epoch {i + 1}")
        train_epoch(model, train_loader, optimizer)
        test(model, test_loader)


def run() -> None:
    should_train = True  # set to True to train the model first
    n_epochs = 1
    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, "model.pt")
    data_dir = os.path.join(dirname, "..", "..", "data")

    train_loader, test_loader = get_data_loaders(data_dir)
    model = load_or_create_model(model_path)

    if should_train:
        train_model(model, n_epochs, train_loader, test_loader)
        torch.save(model, model_path)
        print("Saved model")


if __name__ == "__main__":
    run()
