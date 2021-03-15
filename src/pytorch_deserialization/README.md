# PyTorch Deserialization Attack

This directory demonstrates how to exploit the `torch.load` functionality to run arbitrary code and shows why this method should never be used to load serialized models from an untrusted source. This might be tempting, for example in a federated learning context where a model has to be sent from one user to another.

## Demo

The exploit scenario is this: a victim receives a file from the attacker, which they believe holds a serialized PyTorch model. The victim uses the `torch.load` method in an attempt to use the model but inadvertantly runs the attacker’s arbitrary code.

In this demo, the attacker’s code spawns a new process which prints its PID to the terminal, sleeps for a few seconds, then writes a datestamp to the file `output.tmp`.

The attacker’s code is in `attacker.py` and the victim’s code is in `victim.py`. You can run both with the following script:

```sh
./run.sh
```

## How this attack works

The `torch.save` and `torch.load` methods use Python’s `pickle` module under the hood. As it states in the `pickle` documentation, as well as the docstring for `torch.load`, this should only be used for trusted inputs, because Python classes can define how they should be unpickled.

## How to defend against this attack

Do not use `torch.load` on untrusted inputs. If you need to load model data from untrusted sources you should use a secure serialization format, such as JSON.
