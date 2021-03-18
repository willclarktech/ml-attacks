dependencies = ["torch"]
import subprocess

from torch import nn

ANSI_RED_BOLD = "\033[31;1m"
ANSI_RESET = "\033[0m"

output_filename = "output.tmp"
echo_code = f'echo "{ANSI_RED_BOLD}Spawned subprocess with PID: $${ANSI_RESET}"'
sleep_code = "sleep 5"
write_code = f'echo "Hello at `date`" >> {output_filename}'
example_payload = f"{echo_code}; {sleep_code}; {write_code}"


def my_pretrained_model() -> nn.Module:
    subprocess.Popen(
        (
            "sh",
            "-c",
            example_payload,
        )
    )
    # Return some functional model
    return nn.Linear(2, 1)
