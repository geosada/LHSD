import torch
from torch.utils.data import Dataset

class SimpleTensorDataset(Dataset):
    """
    Returns a single Tensor per __getitem__ (no tuple), shape (C,H,W).
    Keeps data on CPU; your trainer should move to device per batch.
    """
    def __init__(self, x: torch.Tensor):
        assert x.ndim == 4, "Expect (N,C,H,W)"
        self.x = x.contiguous().float()  # keep CPU for pin_memory

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]
