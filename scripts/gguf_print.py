import logging
import sys
from pathlib import Path

from gguf.gguf_reader import GGUFReader

logger = logging.getLogger("reader")


def read_gguf_file(gguf_file_path):
    reader = GGUFReader(gguf_file_path)

    # List all key-value pairs in a columnized format
    print("Key-Value Pairs:")  # noqa: NP100
    max_key_length = max(len(key) for key in reader.fields.keys())
    for key, field in reader.fields.items():
        value = field.parts[field.data[0]]
        print(f"{key:{max_key_length}} : {value}")  # noqa: NP100
    print("----")  # noqa: NP100

    # List all tensors
    print("Tensors:")  # noqa: NP100
    tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Quantization: {}"
    print(tensor_info_format.format("Tensor Name", "Shape", "Size", "Quantization"))  # noqa: NP100
    print("-" * 80)  # noqa: NP100
    for tensor in reader.tensors:
        shape_str = "x".join(map(str, tensor.shape))
        size_str = str(tensor.n_elements)
        quantization_str = tensor.tensor_type.name
        print(
            tensor_info_format.format(
                tensor.name, shape_str, size_str, quantization_str
            )
        )  # noqa: NP100


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.info("Default: /mnt/data/gguf/DeepSeek-R1-Q4_K_M.gguf")
        gguf_file_path = Path("/mnt/data/gguf/DeepSeek-R1-Q4_K_M.gguf")
    else:
        gguf_file_path = Path(sys.argv[1])
    if not gguf_file_path.is_file():
        logger.error(f"File not found: {gguf_file_path}")
        sys.exit(1)
    read_gguf_file(gguf_file_path)
