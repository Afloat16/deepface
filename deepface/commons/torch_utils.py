"""Helpers used only by the optional PyTorch backend."""

from typing import Any, cast

import torch
from numpy.typing import NDArray


def to_numpy(tensor: torch.Tensor) -> NDArray[Any]:
    """Convert an inference output, widening bfloat16 to a NumPy-supported dtype."""
    # Autocast may return bfloat16 even when model weights and inputs are float32.
    # NumPy has no native bfloat16 dtype; other supported dtypes stay unchanged.
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return cast(NDArray[Any], tensor.cpu().numpy())
