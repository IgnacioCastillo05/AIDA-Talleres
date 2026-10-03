import json, os, platform, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "hardware_profile.json"

def ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        return None

def nvidia_info():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    try:
        out = subprocess.check_output([
            exe, "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits"
        ], text=True, stderr=subprocess.DEVNULL, timeout=8).strip().splitlines()[0]
        name, mem = [x.strip() for x in out.split(",", 1)]
        return name, round(float(mem) / 1024.0, 1)
    except Exception:
        return None, None

def torch_info():
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        if cuda:
            name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory/(1024**3), 1)
        else:
            name, vram = None, None
        return torch.__version__, cuda, name, vram
    except Exception:
        return None, False, None, None

ram = ram_gb()
ngpu, nvram = nvidia_info()
torch_ver, torch_cuda, tgpu, tvram = torch_info()
usable_vram = tvram if torch_cuda else None

# Profiles are based on hardware that PyTorch can actually use.
if usable_vram is None:
    profile = "LOW"
elif usable_vram >= 12:
    profile = "STRONG"
elif usable_vram >= 7:
    profile = "STANDARD"
elif usable_vram >= 3:
    profile = "BASIC"
else:
    profile = "LOW"

info = {
    "profile": profile,
    "python": sys.version.split()[0],
    "os": platform.platform(),
    "cpu": platform.processor() or platform.machine(),
    "cpu_logical_cores": os.cpu_count(),
    "ram_gb": ram,
    "nvidia_gpu_detected": ngpu,
    "nvidia_vram_gb": nvram,
    "torch_version": torch_ver,
    "torch_cuda_available": torch_cuda,
    "torch_gpu": tgpu,
    "torch_vram_gb": tvram,
}
OUT.write_text(json.dumps(info, indent=2), encoding="utf-8")

print("\nCOMPUTER SUMMARY")
print("----------------")
for k,v in info.items():
    print(f"{k:24}: {v}")
print("\nAssigned training profile:", profile)
if ngpu and not torch_cuda:
    print("\nNOTE: An NVIDIA GPU was detected, but this PyTorch installation cannot use CUDA.")
    print("The workshops will safely use the LOW profile until CUDA-enabled PyTorch is installed.")
print("\nThe workshops read hardware_profile.json automatically. Do not edit it during the class.")
