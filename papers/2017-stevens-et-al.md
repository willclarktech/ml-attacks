# Stevens et al (2017) Summoning Demons

https://arxiv.org/abs/1701.04739

## tl;dr

They fuzzed some ML libraries and found some serious bugs.

## Summary

-   Look at bugs in ML implementations: “such attacks can be more powerful than traditional adversarial machine learning techniques. For example, a memory corruption vulnerability could allow an adversary to corrupt the entire feature matrix, not just the entries that correspond to adversary-controlled inputs”.
-   Types of exploits: mispredictions, DoS, code execution.
-   “In searching for exploitable bugs, the adversary does not pursue the usual goals of vulnerability exploitation, e.g., gaining control over remote hosts, achieving privilege escalation, escaping sandboxes, etc. Instead, the adversary’s goal is to corrupt the outputs of machine learning programs using silent failures.”
-   Steered fuzzing (semi-automated technique for exploring attack surface): distinguish between crashes and misclassifications by adjusting traditional fuzzing methods (American Fuzzy Lop tool).
-   Found seven vulnerabilities, including three CVEs.
-   “For some of the bugs that we discovered, it is unclear who is responsible for fixing them”.

## Relevance to PPML

Code execution could lead to data leakage.

## Potential demo

Show how to fuzz ML models the way these researchers did?
