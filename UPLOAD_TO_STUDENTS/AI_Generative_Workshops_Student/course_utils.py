from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
DATA_DIR.mkdir(exist_ok=True)

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_output(workshop_folder):
    out = Path(workshop_folder) / "outputs"
    ckpt = Path(workshop_folder) / "checkpoints"
    out.mkdir(exist_ok=True); ckpt.mkdir(exist_ok=True)
    return out, ckpt

def fashion_loader(batch_size=64, max_train=8000, train=True, gan=False, test_mode=False):
    """Return FashionMNIST; --test mode uses synthetic tensors and never downloads data."""
    if test_mode:
        n = min(128, max_train or 128)
        x = torch.rand(n, 1, 28, 28)
        if gan: x = x * 2 - 1
        y = torch.zeros(n, dtype=torch.long)
        return DataLoader(TensorDataset(x, y), batch_size=min(batch_size, n), shuffle=True)
    tfm = [transforms.ToTensor()]
    if gan:
        tfm.append(transforms.Normalize((0.5,), (0.5,)))
    try:
        ds = datasets.FashionMNIST(DATA_DIR, train=train, download=True, transform=transforms.Compose(tfm))
    except Exception as exc:
        raise RuntimeError(
            "FashionMNIST is not available. Connect to the internet once so torchvision can download it "
            f"into {DATA_DIR}, then run again. Original error: {exc}"
        ) from exc
    if max_train and len(ds) > max_train:
        ds = Subset(ds, range(max_train))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=0)

def save_tensor_grid(images, path, title=None, value_range=None, nrow=8, show=True):
    images = images.detach().cpu()
    grid = utils.make_grid(images, nrow=nrow, normalize=value_range is not None, value_range=value_range)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(grid.permute(1,2,0).squeeze(), cmap='gray')
    ax.axis('off')
    if title: ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150)
    if show: plt.show()
    plt.close(fig)
