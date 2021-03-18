# PyTorch Hub Loader Attack

This directory demonstrates how the `torch.hub.load` method can be exploited by a malicious model provider. It shows why this method should never be used to load a model from an untrusted source. The attacker code is in the `hubconf.py` file at the root level of this repository.

## Scenario

A victim is told that a pretrained model is available for their use in a GitHub repository. The victim uses the `torch.hub.load` method to download the model and use it in a Python script. As part of the loading process, PyTorch runs the attacker’s arbitrary code defined in the `hubconf.py` file of the GitHub repository.

## Demo

In this demo, the model provider’s code spawns a new process on the victim’s machine which prints its PID to the terminal, sleeps for a few seconds, then writes a datestamp to the file `output.tmp`.

The victim’s code is in `victim.py` and can be run with the following:

```sh
python victim.py
```

## How this attack works

The `torch.hub.load` method downloads the GitHub repository and then runs the specified endpoint function which is defined in the `hubconf.py` file. The model provider can put any code here and it will run on the victim’s machine.

## How to defend against this attack

Do not use `torch.hub.load` on untrusted inputs.
