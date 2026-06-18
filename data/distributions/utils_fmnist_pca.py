import torch
from torch.utils.data import DataLoader, Subset
try:
    from torchvision import datasets, transforms
except ImportError as e:
    raise ImportError(
        "This helper requires torchvision. Install with `pip install torchvision`."
    ) from e


@torch.no_grad()
def fit_fmnist_pca(n_samples=5000, target_class=7, K=32, seed=42, root="./data"):
    from torchvision import datasets, transforms
    import torch

    g = torch.Generator()
    g.manual_seed(int(seed))

    ds = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )

    idxs = (ds.targets == int(target_class)).nonzero(as_tuple=True)[0]
    n = min(n_samples, idxs.numel())
    perm = torch.randperm(idxs.numel(), generator=g)[:n]
    X = torch.stack([ds[i][0].view(-1) for i in idxs[perm]], dim=0)  # (n, 784)

    mu = X.mean(dim=0)                 # (784,)
    Xc = X - mu

    U_l, S, V_r = torch.pca_lowrank(Xc, q=K)  # Xc ≈ U_l @ diag(S) @ V_r.T
    U_img = V_r[:, :K].contiguous()           # (784, K)  ← use this as image basis

    return mu, U_img

