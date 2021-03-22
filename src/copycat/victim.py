import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

vocab = [
    "aardvark",  # Not part of training set
    "black",
    "blue",
    "cat",
    "dog",
    "green",
    "horse",
    "pig",
    "red",
    "sheep",
    "white",
    "yellow",
]
input_width = len(vocab)

private_training_data = [
    ("black", 0),  # Misclassified
    ("blue", 1),
    ("cat", 0),
    ("dog", 0),
    ("green", 1),
    ("horse", 0),
    ("pig", 0),
    ("red", 1),
    ("sheep", 0),
    ("white", 1),
    ("yellow", 1),
]
training_inputs = F.one_hot(
    torch.as_tensor([vocab.index(word) for word, _ in private_training_data]),
    num_classes=input_width,
).float()
training_targets = torch.as_tensor(
    [[label] for _, label in private_training_data]
).float()

# Train a binary classifier for colour words vs animal words
model = nn.Sequential(nn.Linear(input_width, 1), nn.Sigmoid())
optimizer = optim.SGD(model.parameters(), lr=10)

for _ in range(100):
    optimizer.zero_grad()
    output = model(training_inputs)
    loss = F.binary_cross_entropy(output, training_targets)
    loss.backward()
    optimizer.step()

print(f"Trained model with final training loss {loss:.3g}")


# Deploy the model (in this example we simply save/load it)
model_filename = "model.pt"
torch.save(model, model_filename)
