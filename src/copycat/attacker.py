import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

# The input space is publicly known
vocab = [
    "aardvark",  # Not part of training set
    "black",  # Misclassified in original training set
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

# Access the deployed model eg via a webserver
# In this example we simply load the model file
model_filename = "model.pt"
victim_model = torch.load(model_filename).eval()
# Skip "aardvark" for now (see below)
attack_inputs = F.one_hot(torch.as_tensor([*range(1, input_width)])).float()

# Extract outputs for the training set
extracted_outputs = victim_model(attack_inputs).detach()

# Create a fresh model with a plausible architecture
hidden_width = 8
reconstructed_model = nn.Sequential(
    nn.Linear(input_width, hidden_width),
    nn.Tanh(),
    nn.Linear(hidden_width, 1),
    nn.Sigmoid(),
)
optimizer = optim.Adam(reconstructed_model.parameters(), lr=0.1)

# Train the model to reconstruct the behavior of the victim’s model
for _ in range(100):
    optimizer.zero_grad()
    output = reconstructed_model(attack_inputs)
    loss = F.binary_cross_entropy(output, extracted_outputs)
    loss.backward()
    optimizer.step()

print(f"Reconstructed model behaviour with final training loss {loss:.3g}")
# From original training set
true_labels = torch.as_tensor([0.0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1]).unsqueeze(dim=-1)
true_loss = F.binary_cross_entropy(output, true_labels)
print(
    f"Reconstructed model behaviour true loss against original training set: {true_loss:.3g}"
)

key = "(0 = animal; 1 = color)"

# Manually inspect input-output pairs to discover mistakes
print(f'Found misclassified input: "black"')
black_input = F.one_hot(torch.as_tensor([1]), num_classes=input_width).float()
black_output = reconstructed_model(black_input).item()
print(f"Reconstructed model output: {black_output:.3g} {key}")
# We can now run it on the deployed model for malicious purposes
victim_black_output = victim_model(black_input).item()
print(f"Victim output: {victim_black_output:.3g} {key}")

# Surreptitiously test unseen inputs using reconstructed model
print(f'Trying unseen input "aardvark" on reconstructed model')
aardvark_input = F.one_hot(torch.as_tensor([0]), num_classes=input_width).float()
aardvark_output = reconstructed_model(aardvark_input).item()
print(f"Reconstructed model output: {aardvark_output:.3g} {key}")
# If the input was misclassified, we can now run it on the deployed model for malicious purposes
victim_aardvark_output = victim_model(aardvark_input).item()
print(f"Victim output: {victim_aardvark_output:.3g} {key}")
