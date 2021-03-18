from torch import hub

loaded_model = hub.load(
    "willclarktech/ml-attacks:main", "my_pretrained_model", force_reload=True
)

print("Loaded model:")
print(loaded_model)
