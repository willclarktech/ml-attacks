# Slack et al (2020) Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods

https://arxiv.org/abs/1911.02508

## tl;dr

Methods for generating post-hoc explanations of ML black boxes are increasingly popular for interpretability. However, there has been little analysis of their robustness, especially in adversarial settings. The authors develop a framework for creating biased models which yield innocuous-seeming explanations using two popular techniques.

## Summary

-   LIME and SHAP try to provide explanations of black box models by providing a simple (linear) model approximation for the relevant input’s locality.
-   They do this by perturbing inputs to generate neighbouring inputs, but the distribution of perturbed inputs can be distinguished from the distribution of original inputs.
-   This knowledge can be used to generate a classifier which acts with arbitrary bias, by behaving differently on inputs in the original distribution versus inputs from the perturbed distribution.
-   Experiments with multiple datasets (eg recidivism risk, credit scoring etc) show effectiveness of technique.
-   Nice overview of (widely) related work with references.

## Relevance to PPML

Not really relevant.

## Potential demo

Create a biased model which tricks LIME/SHAP into making it look unbiased.
