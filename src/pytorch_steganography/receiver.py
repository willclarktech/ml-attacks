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
    plaintext = bytes(
        "Stegosaurus is a genus of herbivorous thyreophoran dinosaur.",
        "utf-8",
    )
    dirname = os.path.dirname(__file__)
    original_model_path = os.path.join(dirname, "models", "original.pt")
    modified_model_path = os.path.join(dirname, "models", "modified.pt")
    original_model = torch.load(original_model_path)
    modified_model = torch.load(modified_model_path)
    data_dir = os.path.join(dirname, "..", "..", "data")

    _, test_loader = get_data_loaders(data_dir)

    print("Testing original model...")
    test(original_model, test_loader)
    print("Testing modified model...")
    test(modified_model, test_loader)

    reconstructed = extract_data(modified_model, len(plaintext))
    print("Original plaintext:\t\t", plaintext.decode("utf-8"))
    print("Reconstructed plaintext:\t", reconstructed.decode("utf-8"))


if __name__ == "__main__":
    run()
