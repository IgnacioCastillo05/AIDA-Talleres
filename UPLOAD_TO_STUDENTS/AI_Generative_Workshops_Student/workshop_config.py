import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
PROFILE_FILE = ROOT / "hardware_profile.json"

SETTINGS = {
    "LOW": {
        "autoencoder": {"batch_size": 64, "epochs": 2, "max_train": 8000, "latent_dim": 16},
        "dcgan": {"batch_size": 64, "epochs": 1, "max_train": 8000, "nz": 64, "base": 32},
        "ddpm": {"batch_size": 32, "epochs": 1, "max_train": 4000, "n_steps": 80, "width": 24},
        "flow_matching": {"batch_size": 256, "iterations": 1000, "hidden": 64, "sample_steps": 40},
    },
    "BASIC": {
        "autoencoder": {"batch_size": 96, "epochs": 3, "max_train": 16000, "latent_dim": 24},
        "dcgan": {"batch_size": 96, "epochs": 2, "max_train": 16000, "nz": 64, "base": 48},
        "ddpm": {"batch_size": 48, "epochs": 2, "max_train": 8000, "n_steps": 120, "width": 32},
        "flow_matching": {"batch_size": 512, "iterations": 2000, "hidden": 96, "sample_steps": 60},
    },
    "STANDARD": {
        "autoencoder": {"batch_size": 128, "epochs": 4, "max_train": 30000, "latent_dim": 32},
        "dcgan": {"batch_size": 128, "epochs": 3, "max_train": 30000, "nz": 96, "base": 64},
        "ddpm": {"batch_size": 64, "epochs": 2, "max_train": 16000, "n_steps": 160, "width": 40},
        "flow_matching": {"batch_size": 768, "iterations": 3500, "hidden": 128, "sample_steps": 80},
    },
    "STRONG": {
        "autoencoder": {"batch_size": 256, "epochs": 5, "max_train": 60000, "latent_dim": 48},
        "dcgan": {"batch_size": 192, "epochs": 4, "max_train": 60000, "nz": 128, "base": 64},
        "ddpm": {"batch_size": 96, "epochs": 3, "max_train": 30000, "n_steps": 200, "width": 48},
        "flow_matching": {"batch_size": 1024, "iterations": 5000, "hidden": 192, "sample_steps": 100},
    },
}

def profile_name():
    if PROFILE_FILE.exists():
        try:
            return json.loads(PROFILE_FILE.read_text(encoding="utf-8")).get("profile", "LOW")
        except Exception:
            pass
    return "LOW"

def get_config(workshop):
    p = profile_name()
    return dict(SETTINGS.get(p, SETTINGS["LOW"])[workshop], profile=p)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
