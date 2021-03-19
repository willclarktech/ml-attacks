from torch import hub

# This commit is on a fork controlled by the attacker.
# A GitHub feature associates it with the original repository.
# See https://github.com/huggingface/transformers/blob/3276c70/hubconf.py#L85-L87
source = "huggingface/transformers:3276c70f4"
model = hub.load(source, "model", "bert-base-uncased")
print("Loaded model:")
print(model)
