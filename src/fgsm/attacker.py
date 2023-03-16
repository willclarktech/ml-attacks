from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
# White box attack: we know about the model and the data
from victim import get_data_loaders, load_or_create_model

Example = tuple[float, float, Tensor]


def perturb_image(image: Tensor, gradient: Tensor, epsilon: float) -> Tensor:
    perturbed_image = image + epsilon * gradient.sign()
    return perturbed_image.clamp(0, 1)


def attack(model: nn.Sequential, loader: DataLoader, epsilon: float) -> tuple[float, list[Example]]:
    correct = 0
    examples: list[Example] = []

    for inp, target in loader:
        inp.requires_grad = True
        initial_output = model(inp)
        initial_prediction = initial_output.max(1, keepdim=True)[1]

        # Bail if model doesn't predict the correct output anyway
        if initial_prediction.item() != target.item():
            continue

        loss = F.nll_loss(initial_output, target)
        model.zero_grad()
        loss.backward()

        perturbed_image = perturb_image(inp, inp.grad.data, epsilon)
        final_output = model(perturbed_image)
        final_prediction = final_output.max(1, keepdim=True)[1]

        if final_prediction.item() == target.item():
            correct += 1
            # Special case for epsilon = 0
            if len(examples) < 5 and epsilon == 0:
                example = perturbed_image.squeeze().detach().numpy()
                examples.append((initial_prediction.item(), final_prediction.item(), example))
        else:
            if len(examples) < 5:
                example = perturbed_image.squeeze().detach().numpy()
                examples.append((initial_prediction.item(), final_prediction.item(), example))

    accuracy = correct/float(len(loader))
    return accuracy, examples


def plot_accuracy(epsilons: list[float], accuracies: list[float]) -> None:
    plt.figure()
    plt.plot(epsilons, accuracies)
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.xlabel("Accuracy")
    plt.show()


def plot_examples(epsilons, examples) -> None:
    plt.figure(figsize=(8, 10))
    count = 0

    for i, exs in enumerate(examples):
        for j, example in enumerate(exs):
            count += 1
            plt.subplot(len(examples), len(exs), count)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps {epsilons[i]}")
            original, final, ex = example
            plt.title(f"{original} -> {final}")
            plt.imshow(ex, cmap="gray")

    plt.tight_layout()
    plt.show()


def run() -> None:
    epsilons = np.arange(0.0, 1.0, 0.05)
    dirname = os.path.dirname(__file__)
    model_path = os.path.join(dirname, "model.pt")
    data_dir = os.path.join(dirname, "..", "..", "data")

    _, test_loader = get_data_loaders(data_dir)
    model = load_or_create_model(model_path)
    model.eval()

    accuracies: list[float] = []
    examples: list[list[Example]] = []
    for epsilon in epsilons:
        accuracy, exs = attack(model, test_loader, epsilon)
        print(f"Epsilon {epsilon}: accuracy = {accuracy}")
        accuracies.append(accuracy)
        examples.append(exs)

    plot_accuracy(epsilons, accuracies)
    plot_examples(epsilons, examples)




if __name__ == "__main__":
    run()
