# Xiao et al (2017) Security Risks in Deep Learning Implementations

https://arxiv.org/abs/1711.11008

## tl;dr

Deep learning frameworks are great but (1) they’re very large (100,000s lines of code) and (2) they rely on numerous 3rd party libraries. Implementation complexity often leads to security vulnerabilities (and does so here). These include DoS, misclassification, system compromise.

## Summary

-   Demonstrates complexity of Caffe/TensorFlow/Torch with lines of code/number of dependencies.
-   Nice summary of attack surface for a deep learning application: inputs, training data, shared models.
-   Researchers found dozens of vulnerabilities, 15 of which were assigned CVE numbers.
-   Mostly DoS/crash.
-   “For software bugs that allows an attacker to hijack control flow, attackers can potentially leverage the software bug and remotely compromise the system that hosts deep learning applications.”
-   They used traditional methods like fuzzing to find the vulnerabilities, although there are some limitations for using these on deep learning applications.

## Relevance to PPML

It doesn’t matter how fancy your privacy protocol is or how small your epsilon budget if a deserialisation bug in TensorFlow gives an attacker RCE on the machine with the data.

## Potential demo

Use a vulnerable version of some framework and show how to exploit it.
