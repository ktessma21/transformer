

import numpy.typing as npt
import numpy as np
import torch
import os 
import typing 


def save_checkpoint(
    model : torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iter: int, 
    out : str | os.PathLike | typing.BinaryIO |  typing.IO[bytes]
) -> None:

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iter,
        },
        out,
    )


def load_checkpoint(
    src : str | os.PathLike | typing.BinaryIO |  typing.IO[bytes],
    model : torch.nn.Module, 
    optimizer: torch.optim.Optimizer
) -> int:
    
    
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]


def load_data(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:

    starts = np.random.randint(
        0, len(x) - context_length, size=batch_size
    )
    inputs = []
    targets = []

    for i in starts:
        inputs.append(x[i : i + context_length])
        targets.append(x[i + 1 : i + context_length + 1])
    
    inputs = torch.as_tensor(
        np.stack(inputs).astype(np.int32), dtype=torch.int32, device=device
    )

    targets = torch.as_tensor(
        np.stack(targets).astype(np.int32), dtype=torch.int32, device=device
    )

    return (inputs, targets)
