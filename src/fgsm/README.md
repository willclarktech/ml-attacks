# Adversarial Examples using the Fast Gradient Sign Method (FGSM)

This directory demonstrates how to construct an FGSM attack to create adversarial examples for an MNIST image classifier. The code is heavily inspired by [a PyTorch tutorial](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html) but the neural network used here is simpler.

## Demo

In this demo, the victim trains a model on a dataset and saves it to a file. Since FGSM is a white box technique, the attacker has access to the trained model as well as the test dataset. The attacker selects input images from the test dataset and perturbs them to find adversarial examples which get misclassified by the model.

The victim’s code is in `victim.py` and the attacker’s code is in `attacker.py`. You can run both with the following script:

```sh
./run.sh
```

## How this attack works

FGSM is a white box technique with the goal of misclassification. The attacker uses a crude form of gradient ascent to perturb a chosen input image in the direction of increased loss. It is "fast" because each data point in the image is adjusted by the same amount (controlled by the parameter `epsilon`), so no scaling is performed eg according to how each data point contributes to the loss. By progressively increasing epsilon the attacker can find a perturbation which will change the classification, and (depending on the image, the model, and the size of epsilon) in many cases the perturbed image will look very similar to the original.

## How to defend against this attack

Szegedy et al (2014) suggest feeding adversarial examples back into the training set. This is essentially a data augmentation technique.

## Resources

-   [Szegedy et al (2014) Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199v4): Describes adversarial examples and how to generate them using a loss function
-   [Goodfellow et al (2015) Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572v3): Describes FGSM
