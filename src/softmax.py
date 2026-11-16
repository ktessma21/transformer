
from typing import Any

import torch 

from torch import nn

def softmax(x : torch.Tensor, dim :int ):
    # now bigger concern is that we might have elemnt from torch with 
    max_vals,_  = x.max(dim=dim, keepdim=True)

    shifted = x - max_vals

    exps = torch.exp(shifted)
    return exps / exps.sum(dim=dim, keepdim=True)

     