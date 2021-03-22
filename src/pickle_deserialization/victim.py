import os
import pickle

dirname = os.path.dirname(__file__)
malicious_model_filename = os.path.join(dirname, "malicious_model.pt")

with open(malicious_model_filename, "rb") as malicious_model_file:
    malicious_model = pickle.load(malicious_model_file)

print("Loaded model:")
print(malicious_model)
