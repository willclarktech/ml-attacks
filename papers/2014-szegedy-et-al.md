# Szegedy et al (2014) Intriguing properties of neural networks

https://arxiv.org/abs/1312.6199v4

## tl;dr

This paper introduced the idea of adversarial inputs for deep neural networks, and an efficient technique for generating them.

## Summary

-   Deep neural networks are powerful but hard to interpret and have counterintuitive properties.
-   Previous work analyzed the semantic meaning of individual units by finding the set of inputs that maximally activate a given unit.
-   The researchers show that "there is no distinction between individual high level units and random linear combinations of high level units, [which] suggests that it is the space, rather than the individual units, that contains the semantic information in the high layers of neural networks."
-   The researchers specify a technique for finding adversarial examples which produce imperceptibly different inputs which a deep neural network will misclassify.
-   These adversarial examples also show cross-model generalization (work on models trained from scratch with different hyperparameters) as well as cross-training set generalization (for models trained from scratch on a different training set).
-   Feeding adversarial examples back into the training set can help produce more robust models.

## Relevance to PPML

Not really relevant.

## Potential demo

Reproduce adversarial examples for MNIST.
