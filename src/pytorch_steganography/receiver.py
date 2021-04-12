import os

import torch
from torch import nn

from sender import float_to_int, get_data_loaders, test


BITS_PER_BYTE = 8


def bits_to_bytes(bits: list[int]) -> bytearray:
    n_bytes = len(bits) // BITS_PER_BYTE
    data = bytearray(n_bytes)

    for k in range(n_bytes):
        for l in range(BITS_PER_BYTE):
            data[k] <<= 1
            data[k] |= bits[k * 8 + l]

    return data


def extract_data(model: nn.Sequential, n_bytes: int) -> bytearray:
    n_bits = n_bytes * BITS_PER_BYTE
    i = 0
    n_parameters = 0
    bits = []
    for params in model.parameters():
        flat_params = params.flatten()
        n_parameters += len(flat_params)

        for param in flat_params:
            param = float_to_int(param.item())
            bits.append(param & 1)

            i += 1
            if i >= n_bits:
                break

        if i >= n_bits:
            break

    if i < n_bits:
        raise ValueError(
            f"Not enough model parameters ({n_parameters}) for data bits ({n_bits})"
        )

    return bits_to_bytes(bits)


def run() -> None:
    n_bytes = 22_566
    dirname = os.path.dirname(__file__)
    original_model_path = os.path.join(dirname, "models", "original.pt")
    modified_model_path = os.path.join(dirname, "models", "modified.pt")
    original_model = torch.load(original_model_path)
    modified_model = torch.load(modified_model_path)
    data_dir = os.path.join(dirname, "..", "..", "data")
    original_file_name = "stegosaurus.png"
    original_file_path = os.path.join(dirname, original_file_name)
    reconstructed_file_name = "reconstructed.png"
    reconstructed_file_path = os.path.join(dirname, reconstructed_file_name)

    _, test_loader = get_data_loaders(data_dir)

    print("Testing original model...")
    test(original_model, test_loader)
    print("Testing modified model...")
    test(modified_model, test_loader)

    original_file = open(original_file_path, "rb").read()
    n_bytes = len(original_file)  # This could be transmitted via the model as a varint
    reconstructed = extract_data(modified_model, n_bytes)
    with open(reconstructed_file_path, "wb") as reconstructed_file:
        reconstructed_file.write(reconstructed)

    successfully_reconstructed = reconstructed == original_file
    message = (
        f"{reconstructed_file_name} successfully reconstructed {original_file_name}"
        if successfully_reconstructed
        else f"{reconstructed_file_name} did not successfully reconstruct {original_file_name}"
    )
    print(message)


if __name__ == "__main__":
    run()
