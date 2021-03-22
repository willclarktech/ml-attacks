# Copycat Attack

This directory demonstrates how to construct a copycat attack. It is based on the first CVE assigned to a machine learning bug: [CVE-2019-20634](https://nvd.nist.gov/vuln/detail/CVE-2019-20634) for the Proofpoint spam email filter system.

## Scenario

The victim deploys a machine learning model such that an attacker can cheaply submit inputs and receive the outputs of the model. The attacker uses this access to reconstruct the model. The reconstructed model is then used for malicious purposes, which could be as simple as theft of intellectual property (the model). In the case of Proofpoint, the attacker used the reconstructed model to find spam emails which would evade the filters.

## Demo

In this demo, the victim trains a model on a private dataset and "deploys" it by saving the model to a file. The attacker accesses the "deployment" by loading the file, and uses the outputs of the model to duplicate it and find misclassifications which could potentially be exploited.

The victim’s code is in `victim.py` and the attacker’s code is in `attacker.py`. You can run both with the following script:

```sh
./run.sh
```

## How this attack works

The deployed model accepts inputs from the attacker and passes its outputs back to the attacker, who can use the input-output pairs to form a training dataset for a new model designed to mimic the original model’s behavior. The training set as well as the reconstructed model can be inspected to find potential misclassifications.

## How to defend against this attack

Make it expensive for the attacker to obtain outputs from your model and/or pass back only indirect outputs. For example, the Proofpoint attackers were able to send emails costlessly to acquire a dataset, and detailed information about Proofpoint’s model’s outputs was provided in the metadata of the emails which were received, allowing the attackers to gain important insights into the structure of the model. If it had cost money to send the emails the attackers may have been dissuaded, and if less detailed outputs had been returned it would have been harder to reconstruct the model.
