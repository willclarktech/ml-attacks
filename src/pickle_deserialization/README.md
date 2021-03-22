# Pickle Deserialization Attack

This directory demonstrates how to use the [fickling](https://github.com/trailofbits/fickling) library to generate malicious pickle files. This tool can also be used to analyse pickle files and detect certain warning signs.

## Scenario

A victim unpickles some data from a file which has been created (or modified) by the attacker. For example, this could be a model file which the victim wants to use. This attack can be combined with the [PyTorch deserialization attack](https://github.com/willclarktech/ml-attacks/tree/main/src/pytorch_deserialization) to create valid model files, making the malicious execution more difficult to detect.

## Demo

In this demo, the attacker’s code simply prints some red text to the terminal, but this could be modified to run arbitrary Python code.

The attacker’s code is in `attacker.py` and the victim’s code is in `victim.py`. You can run both with the following script:

```sh
./run.sh
```

## How this attack works

Pickle files are actually compiled programs which, when unpickled, run in a special virtual machine. Although this virtual machine is limited in certain ways, but can still run arbitrary code. A deeper explanation is available [here](https://blog.trailofbits.com/2021/03/15/never-a-dill-moment-exploiting-machine-learning-pickle-files/) from the creators of fickling.

## How to defend against this attack

Do not unpickle untrusted data.
