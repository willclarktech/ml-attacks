import os

import torch

dirname = os.path.dirname(os.path.realpath(__file__))
model_filename = os.path.join(dirname, "model.pt")
default_input = torch.Tensor([[0, 1]])

# Unsafely deserialize the model
# This should spawn a subprocess which prints to the terminal,
# sleeps until the current process has exited, then writes to the output file
model = torch.load(model_filename)
print("Loaded model:")
print(model)
