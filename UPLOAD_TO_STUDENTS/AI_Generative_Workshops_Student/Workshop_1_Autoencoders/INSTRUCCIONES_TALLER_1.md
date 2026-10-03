# Taller 1 — Autoencoders · Guía de ejecución para Claude Code

> **Qué es este archivo:** instrucciones paso a paso para resolver el *Workshop 1 – Autoencoders* (curso AI Generative Workshops) desde Claude Code en la terminal. Está construido a partir del PDF/Word del taller (`Docs/W1_Autoencoder.docx`) y de los archivos de apoyo del repositorio (`workshop_config.py`, `course_utils.py`, `check_computer.py`, `.bat`).
>
> **Validación previa:** el código de los 5 milestones de este documento se extrajo literalmente del `.docx` y se ejecutó con `--test` (datos sintéticos, CPU, PyTorch 2.14): los 5 corrieron sin errores y generaron todos los artefactos esperados. Lo único **no** probado es la descarga real de FashionMNIST (requiere internet).

---

## 0. Cómo usar este archivo

1. Clona/abre el repo y ubícate en la carpeta raíz del proyecto (ver §2).
2. Abre Claude Code ahí (`claude`) y pega este prompt:

   ```text
   Lee Workshop_1_Autoencoders/INSTRUCCIONES_TALLER_1.md completo y ejecútalo en orden.
   Reglas: no avances de milestone hasta que corra bien y yo confirme que entiendo el
   checkpoint. Después de cada milestone detente, resume qué salió (con números y rutas
   de las imágenes) y hazme las preguntas de "Explicar" antes de seguir.
   ```

3. **Rol de Claude Code:** ejecutar, verificar y explicar. **Tu rol:** entender y responder. El taller dice explícitamente *“Do not move to the next milestone until the program runs and you can explain the checkpoint result”* y que las respuestas finales **no** se copian del texto del taller sino de **tus outputs**. Por eso cada milestone termina con un bloque **Explicar** y uno **Think/Change/Test**; úsalos como práctica, no los saltes.

---

## 1. Qué se construye (resumen)

Un **autoencoder convolucional** en PyTorch sobre **FashionMNIST** (sin etiquetas: el target es la propia imagen).

```
x (1×28×28) → Encoder f_θ → z (latent_dim) → Decoder g_φ → x̂ (1×28×28)
```

- `z = f_θ(x)`, `x̂ = g_φ(f_θ(x))`
- Pérdida: `L_MSE = (1/N) Σ ‖xᵢ − x̂ᵢ‖²`
- El latente es un **cuello de botella (bottleneck)**: menos dimensiones ⇒ más compresión; si es demasiado pequeño el decoder no recupera la información.

| Milestone | Archivo a crear | Objetivo | Salidas |
|---|---|---|---|
| 1 | `milestone_01_data.py` | Inspeccionar datos, rango de píxeles y perfil de hardware | `outputs/m1_batch.png` |
| 2 | `milestone_02_model.py` | Construir encoder / latente / decoder y verificar shapes | `outputs/m2_untrained_recon.png` |
| 3 | `milestone_03_train.py` | Entrenar por reconstrucción (MSE + Adam) | `outputs/m3_loss.png`, `outputs/m3_reconstruction.png`, `checkpoints/autoencoder.pt` |
| 4 | `milestone_04_latent_space.py` | Interpolación lineal en el espacio latente | `outputs/m4_latent_interpolation.png` |
| 5 | `milestone_05_bottleneck_experiment.py` | Experimento controlado de `latent_dim` = 4, 16, 64 | `outputs/m5_bottleneck_results.csv`, `outputs/m5_bottleneck.png` |

---

## 2. Estructura real del repositorio (importante)

El proyecto **no** está en la raíz del repo; está anidado:

```
AIDA-Talleres/                                   ← raíz del repo git
└── UPLOAD_TO_STUDENTS/
    └── AI_Generative_Workshops_Student/         ← RAÍZ DEL PROYECTO (aquí viven config y utils)
        ├── Check_Computer.bat / check_computer.py
        ├── Setup_Environment.bat
        ├── requirements.txt
        ├── workshop_config.py                   ← perfiles LOW/BASIC/STANDARD/STRONG
        ├── course_utils.py                      ← fashion_loader, save_tensor_grid, ...
        ├── hardware_profile.json                ← se genera con check_computer.py
        ├── datasets/                            ← FashionMNIST se descarga aquí
        ├── Docs/W1_Autoencoder.docx ...         ← enunciados
        └── Workshop_1_Autoencoders/             ← AQUÍ se crean los 5 milestones
            └── INSTRUCCIONES_TALLER_1.md        ← este archivo
```

Los milestones calculan `ROOT = Path(__file__).resolve().parent.parent` e importan `workshop_config` y `course_utils` desde ahí. **No muevas la carpeta del workshop** fuera de la raíz del proyecto.

> Todos los comandos de abajo asumen que tu directorio actual es la **raíz del proyecto** (`AI_Generative_Workshops_Student/`), salvo que se indique otra cosa.

---

## 3. Avisos de la revisión (leer antes de ejecutar)

1. **Los `.bat` son para doble clic en Windows y terminan en `pause`.** Dentro de Claude Code (no interactivo) se quedarían colgados. Usa los **comandos equivalentes** del §4.
2. **Ventanas de matplotlib bloqueantes.** Fuera de `--test`, los milestones llaman `plt.show()`, que abre una ventana y **detiene el script** hasta cerrarla. En la terminal de Claude Code configura el backend sin ventana:
   - PowerShell: `$env:MPLBACKEND="Agg"`
   - cmd: `set MPLBACKEND=Agg`
   - bash/zsh: `export MPLBACKEND=Agg`

   Las imágenes igual se guardan en `outputs/`; ábrelas desde el explorador de archivos.
3. **GPU NVIDIA en Windows.** `pip install -r requirements.txt` suele instalar PyTorch **solo CPU**; `check_computer.py` lo detecta y asigna perfil **LOW** aunque tengas GPU. Si tienes NVIDIA y quieres un perfil mayor, instala PyTorch con CUDA con el selector oficial (https://pytorch.org/get-started/locally/) y vuelve a correr `check_computer.py`. **No edites `hardware_profile.json` a mano** (lo dice el taller).
4. **Primera ejecución real = descarga de FashionMNIST** a `datasets/` (necesita internet una vez). Con `--test` no se descarga nada (usa tensores aleatorios) — sirve solo como prueba de humo; **las salidas “reales” del taller son las de la ejecución sin `--test`**.
5. **Los THINK/CHANGE/TEST pueden pisar tus resultados.** Ejemplo: el experimento de “doblar el learning rate” en el Milestone 3 sobrescribiría `checkpoints/autoencoder.pt` (que necesita el Milestone 4) y `outputs/m3_loss.png`. Por eso en §5 los experimentos se hacen en **scripts aparte** dentro de `Workshop_1_Autoencoders/experiments/`, guardando con otros nombres. **No edites los milestones oficiales para experimentar**; si lo haces, restaura el valor original (el taller lo pide explícitamente).
6. **`.gitignore`.** El repo no tiene uno. Antes de hacer commit crea uno en la raíz del repo con al menos: `.venv/`, `__pycache__/`, `datasets/`, `hardware_profile.json`, `**/outputs/`, `**/checkpoints/` (o decide conscientemente subir los `outputs/` como evidencia de la entrega).
7. **Semilla fija ⇒ resultados repetibles.** `seed_everything()` usa seed 42; si repites una corrida sin cambiar nada, deberías obtener los mismos números (en CPU; en GPU puede haber diferencias mínimas).

---

## 4. Fase 0 — Preparación del entorno (Checkpoint de arranque)

Equivale a `Setup_Environment.bat` + `Check_Computer.bat`, sin `pause`.

**Windows (PowerShell, desde la raíz del proyecto):**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_computer.py
$env:MPLBACKEND="Agg"
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_computer.py
export MPLBACKEND=Agg
```

> En los comandos siguientes se escribe `python`; en Windows sin activar el venv usa `.\.venv\Scripts\python.exe` en su lugar (o actívalo con `.\.venv\Scripts\Activate.ps1`).

**✅ CHECKPOINT 0** (el taller: *“Do not start Milestone 1 until Check_Computer.bat completes and reports an assigned profile”*):

- Existe `hardware_profile.json` en la raíz del proyecto.
- La terminal imprime `Assigned training profile: LOW|BASIC|STANDARD|STRONG`.
- `python -c "import torch, torchvision, matplotlib; print(torch.__version__)"` funciona.

**Perfiles → hiperparámetros** (de `workshop_config.py`, clave `autoencoder`):

| Perfil | VRAM utilizable | batch_size | epochs | max_train | latent_dim |
|---|---|---|---|---|---|
| LOW | CPU / GPU no usable por PyTorch | 64 | 2 | 8000 | 16 |
| BASIC | ~3–6 GB | 96 | 3 | 16000 | 24 |
| STANDARD | ~7–11 GB | 128 | 4 | 30000 | 32 |
| STRONG | 12+ GB | 256 | 5 | 60000 | 48 |

📝 **Anota en tu reporte** (§7): perfil asignado, dispositivo (`cpu`/`cuda`) y estos 4 hiperparámetros.

---

## 5. Milestones

> **Convención:** crea cada archivo **exactamente con el nombre indicado** en `Workshop_1_Autoencoders/` y con el contenido **literal** de abajo (los comentarios forman parte del material de estudio — el taller pide leerlos). Se puede ejecutar desde la raíz del proyecto (`python Workshop_1_Autoencoders/milestone_01_data.py`) o desde la carpeta del workshop (`python milestone_01_data.py`).
>
> Prueba de humo opcional antes de la corrida real: `python <archivo> --test`.

### Milestone 1 — Inspeccionar datos y perfil de entrenamiento

**Archivo:** `Workshop_1_Autoencoders/milestone_01_data.py`

```python
"""Milestone 1 - Load FashionMNIST, inspect the adaptive profile, and view a batch."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workshop_config import get_config, device
from course_utils import fashion_loader, ensure_output, save_tensor_grid, seed_everything


def main(test=False):
    # Fix random seeds so demonstrations are repeatable across runs.
    seed_everything()

    # The configuration was chosen by Check_Computer.bat from usable hardware.
    # It changes workload, not the mathematical learning objective.
    cfg = get_config("autoencoder")
    out, _ = ensure_output(Path(__file__).parent)

    # test_mode=True uses synthetic images, allowing the instructor smoke test to
    # verify the code without downloading FashionMNIST.
    loader = fashion_loader(
        batch_size=cfg["batch_size"],
        max_train=cfg["max_train"],
        test_mode=test,
    )
    images, labels = next(iter(loader))

    print("Profile:", cfg["profile"], "Device:", device())
    print(
        "Batch shape:", tuple(images.shape),
        "Pixel range:", float(images.min()), float(images.max())
    )

    # FashionMNIST batches must be [batch, channel, height, width].
    assert images.ndim == 4 and images.shape[1:] == (1, 28, 28)

    # Save a visible checkpoint before any neural network is introduced.
    save_tensor_grid(
        images[:32],
        out / "m1_batch.png",
        "Milestone 1 - FashionMNIST batch",
        show=not test,
    )
    print("Saved:", out / "m1_batch.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    main(parser.parse_args().test)
```

**Ejecutar:** `python milestone_01_data.py`

**Debes ver:**
- Terminal: `Profile: <perfil> Device: <cpu|cuda>`, `Batch shape: (batch_size, 1, 28, 28)` y `Pixel range: ~0.0 1.0`.
- `outputs/m1_batch.png` con una grilla de 32 prendas de FashionMNIST.

**Explicar (checkpoint):** por qué las redes reciben tensores `[batch, channel, height, width]` y por qué el taller mantiene los valores de imagen en `[0,1]` (`ToTensor()` divide entre 255; coherente con la `Sigmoid` final del decoder y con la escala de la MSE).

**THINK / CHANGE / TEST**
- *Cambiar cuántas imágenes se muestran sin tocar el batch size:* crea `experiments/m1_show_n.py` (copia de M1) donde el slice `images[:32]` pase a `images[:16]` (y luego `images[:8]`) y guarda en `outputs/m1_batch_16.png`. **Qué cambia:** solo la cantidad de imágenes dibujadas en la figura. **Qué no cambia:** `Batch shape` impreso, `batch_size` del loader (viene de `cfg["batch_size"]`) y el rango de píxeles.
- *Predicción para un tensor `[batch,28,28]`:* `Conv2d(1, 16, ...)` espera `[N,1,28,28]` (o sin batch `[1,28,28]`). Con `[64,28,28]` PyTorch lo interpreta como **imagen sin batch con 64 canales** y falla con *"expected input[1, 64, 28, 28] to have 1 channels, but got 64 channels"*. (Verificado: `[1,28,28]` sí pasaría, pero porque se interpreta como una imagen sin batch.) Solución: `x.unsqueeze(1)`.

**✅ Siguiente paso solo si:** el script corrió, `m1_batch.png` existe y puedes explicar lo anterior.

---

### Milestone 2 — Encoder, cuello de botella y decoder

**Archivo:** `Workshop_1_Autoencoders/milestone_02_model.py`

```python
"""Milestone 2 - Build a convolutional autoencoder and verify tensor shapes."""
import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workshop_config import get_config, device
from course_utils import fashion_loader, ensure_output, save_tensor_grid, seed_everything


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()

        # ENCODER: two stride-2 convolutions reduce spatial size
        # 28x28 -> 14x14 -> 7x7 while increasing the channel count.
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Flatten 32 feature maps of size 7x7 and compress them into z.
        self.to_latent = nn.Linear(32 * 7 * 7, latent_dim)

        # DECODER: first expand z back to a 7x7 feature map.
        self.from_latent = nn.Linear(latent_dim, 32 * 7 * 7)

        # Transposed convolutions reverse the spatial compression:
        # 7x7 -> 14x14 -> 28x28. Sigmoid returns pixels in [0,1].
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        return self.to_latent(h.flatten(1))

    def decode(self, z):
        h = self.from_latent(z).view(-1, 32, 7, 7)
        return self.decoder_conv(h)

    def forward(self, x):
        # An autoencoder is the composition decoder(encoder(x)).
        return self.decode(self.encode(x))


def main(test=False):
    seed_everything()
    cfg = get_config("autoencoder")
    dev = device()
    out, _ = ensure_output(Path(__file__).parent)

    loader = fashion_loader(cfg["batch_size"], cfg["max_train"], test_mode=test)
    x, _ = next(iter(loader))
    x = x.to(dev)

    model = ConvAutoencoder(cfg["latent_dim"]).to(dev)

    # No gradients are required: this milestone checks architecture and shapes.
    with torch.no_grad():
        z = model.encode(x)
        y = model.decode(z)

    print("Input:", tuple(x.shape))
    print("Latent:", tuple(z.shape))
    print("Output:", tuple(y.shape))

    # The decoder must reconstruct the same image shape; z has latent_dim values.
    assert y.shape == x.shape
    assert z.shape[1] == cfg["latent_dim"]

    save_tensor_grid(
        y[:32],
        out / "m2_untrained_recon.png",
        "Untrained autoencoder output",
        show=not test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    main(parser.parse_args().test)
```

**Ejecutar:** `python milestone_02_model.py`

**Debes ver:**
- `Input: (B, 1, 28, 28)`, `Latent: (B, latent_dim)`, `Output: (B, 1, 28, 28)`.
- `outputs/m2_untrained_recon.png`: manchas/ruido sin sentido (red **sin entrenar**).

**Explicar (checkpoint):** cómo las convoluciones con stride 2 reducen la resolución; por qué el decoder debe recuperar exactamente 28×28 (para poder comparar píxel a píxel con la MSE); por qué una red sin entrenar tiene la forma correcta pero ninguna representación útil.

**THINK / CHANGE / TEST**
- *Calcular tamaños espaciales antes de correr* (fórmula `out = ⌊(H + 2p − k)/s⌋ + 1`):
  - Conv1 (k=3, s=2, p=1): `⌊(28+2−3)/2⌋+1 = 14` → `[B,16,14,14]`
  - Conv2 (k=3, s=2, p=1): `⌊(14+2−3)/2⌋+1 = 7` → `[B,32,7,7]`
  - `flatten(1)` → `32·7·7 = 1568` → `Linear(1568, latent_dim)`
  - Decoder, `ConvTranspose2d` (`out = (H−1)·s − 2p + k`, k=4, s=2, p=1): `7→14→28`
- *Cambiar solo `latent_dim`* (por ejemplo `ConvAutoencoder(8)`): solo `to_latent` (salida) y `from_latent` (entrada) dependen de `latent_dim`. **Debe seguir compatible** `32*7*7 = 1568` entre el encoder conv, `to_latent` y `from_latent`/`view(-1,32,7,7)`. Parámetros totales (verificado): `latent_dim=2 → 21 107`, `16 → 65 025`, `24 → 90 121`, `32 → 115 217`, `48 → 165 409`.
- *¿Por qué Sigmoid y no ReLU al final?* Los píxeles están en `[0,1]`: Sigmoid fuerza la salida a ese rango; ReLU no acota por arriba (podría dar >1) y además anula las salidas negativas con gradiente cero (píxeles “muertos”).

**✅ Siguiente paso solo si:** los 3 shapes impresos son correctos y puedes explicar los tamaños 28→14→7→14→28.

---

### Milestone 3 — Entrenar por reconstrucción

**Archivo:** `Workshop_1_Autoencoders/milestone_03_train.py`

> Importa `ConvAutoencoder` desde `milestone_02_model.py`, así que ese archivo debe existir en la misma carpeta.

```python
"""Milestone 3 - Train the autoencoder using reconstruction loss."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workshop_config import get_config, device
from course_utils import fashion_loader, ensure_output, save_tensor_grid, seed_everything
from milestone_02_model import ConvAutoencoder


def main(test=False):
    seed_everything()
    cfg = get_config("autoencoder")
    dev = device()
    out, ckpt = ensure_output(Path(__file__).parent)

    loader = fashion_loader(cfg["batch_size"], cfg["max_train"], test_mode=test)
    model = ConvAutoencoder(cfg["latent_dim"]).to(dev)

    # MSE compares each reconstructed pixel with its original target pixel.
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Smoke tests run one short epoch; normal runs use the hardware profile.
    epochs = 1 if test else cfg["epochs"]
    history = []

    for epoch in range(epochs):
        model.train()
        total_squared_error = 0.0

        for x, _ in loader:
            x = x.to(dev)

            # Standard PyTorch learning cycle: clear -> forward -> loss -> backward -> update.
            optimizer.zero_grad()
            reconstruction = model(x)
            loss = loss_fn(reconstruction, x)
            loss.backward()
            optimizer.step()

            # Weight batch loss by batch size before computing an epoch mean.
            total_squared_error += loss.item() * x.size(0)

        mean_loss = total_squared_error / len(loader.dataset)
        history.append(mean_loss)
        print(f"Epoch {epoch + 1}/{epochs} loss={mean_loss:.5f}")

    # Save enough metadata to reconstruct exactly the same architecture later.
    torch.save(
        {
            "model": model.state_dict(),
            "latent_dim": cfg["latent_dim"],
            "profile": cfg["profile"],
        },
        ckpt / "autoencoder.pt",
    )

    # Plot training behaviour rather than relying on one final loss number.
    fig, ax = plt.subplots()
    ax.plot(range(1, len(history) + 1), history, marker="o")
    ax.set(xlabel="Epoch", ylabel="MSE", title="Autoencoder training loss")
    fig.tight_layout()
    fig.savefig(out / "m3_loss.png", dpi=150)
    plt.close(fig)

    # Compare the same batch before/after the bottleneck visually.
    x, _ = next(iter(loader))
    x = x.to(dev)
    model.eval()
    with torch.no_grad():
        reconstruction = model(x)

    pair = torch.cat([x[:8], reconstruction[:8]], dim=0)
    save_tensor_grid(
        pair,
        out / "m3_reconstruction.png",
        "Top: originals | Bottom: reconstructions",
        nrow=8,
        show=not test,
    )

    if test:
        assert torch.isfinite(torch.tensor(history)).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    main(parser.parse_args().test)
```

**Ejecutar:** `python milestone_03_train.py`

**Debes ver:**
- Líneas `Epoch k/E loss=...` con pérdida **decreciente** (con LOW: 2 épocas; STRONG: 5).
- `outputs/m3_loss.png`, `outputs/m3_reconstruction.png` (arriba originales, abajo reconstrucciones) y `checkpoints/autoencoder.pt`.

📝 **Registra la pérdida final** en el reporte (necesaria para el experimento siguiente).

**Explicar (checkpoint):** por qué el target es igual al input (aprendizaje no supervisado/auto-supervisado: la etiqueta es la propia imagen); qué mide la MSE (error cuadrático medio por píxel entre `x` y `x̂`); por qué `train()` y `eval()` son conceptos distintos aunque este modelo no tenga dropout ni batch-norm (fijan el modo de las capas; es buena práctica y cambia el comportamiento en cuanto se añadan esas capas).

**THINK / CHANGE / TEST — doblar el learning rate (sin pisar tus archivos)**
1. Registra primero la pérdida final de la corrida oficial (lr = `1e-3`).
2. Crea `Workshop_1_Autoencoders/experiments/m3_lr_2x.py` a partir de `milestone_03_train.py` con **estos cambios**: `lr=2e-3`; el `torch.save` apunta a `ckpt / "autoencoder_lr2x.pt"`; las figuras se guardan como `m3_loss_lr2x.png` y `m3_reconstruction_lr2x.png`; ajusta el `sys.path`/`ROOT` (`Path(__file__).resolve().parents[2]`) y agrega `Workshop_1_Autoencoders` al `sys.path` para poder importar `milestone_02_model`.
3. Ejecuta y compara: pérdida final y forma de la curva (¿más ruidosa? ¿diverge? ¿converge más rápido?). Con pocas épocas la diferencia puede ser pequeña: **reporta lo que midas**, no lo que “debería” pasar.
4. **Restaura/no toques** el valor original en el archivo oficial (`lr=1e-3`) y verifica con `git diff` que `milestone_03_train.py` quedó intacto.

*Pregunta de reflexión:* una pérdida de reconstrucción baja **no** prueba que el latente sirva para clasificar — el latente solo está optimizado para **reconstruir** (puede codificar detalles de textura/brillo que no separan clases, o ser casi la identidad si hay capacidad de sobra). Para saberlo hay que **medirlo** con un clasificador sobre `z` (ver pregunta final 4).

**✅ Siguiente paso solo si:** la pérdida baja, existen los 3 artefactos y `autoencoder.pt` es el de la corrida **oficial** (lr=1e-3).

---

### Milestone 4 — Interrogar el espacio latente

**Archivo:** `Workshop_1_Autoencoders/milestone_04_latent_space.py`

> Requiere `checkpoints/autoencoder.pt` del Milestone 3 (si no existe lanza `FileNotFoundError`).

```python
"""Milestone 4 - Interpolate between two latent codes and decode the path."""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workshop_config import device
from course_utils import fashion_loader, ensure_output, save_tensor_grid, seed_everything
from milestone_02_model import ConvAutoencoder


def main(test=False):
    seed_everything()
    dev = device()
    out, ckpt = ensure_output(Path(__file__).parent)
    checkpoint_path = ckpt / "autoencoder.pt"

    # This milestone depends on the trained weights created in Milestone 3.
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Run Milestone 3 first so checkpoints/autoencoder.pt exists."
        )

    saved = torch.load(checkpoint_path, map_location=dev)
    model = ConvAutoencoder(saved["latent_dim"]).to(dev)
    model.load_state_dict(saved["model"])
    model.eval()

    loader = fashion_loader(16, 64, test_mode=test)
    x, _ = next(iter(loader))
    x = x.to(dev)

    with torch.no_grad():
        # Encode two real images to obtain endpoints z0 and z1.
        z = model.encode(x[:2])

        # Nine alpha values give nine points along a straight line in latent space.
        alpha = torch.linspace(0, 1, 9, device=dev).unsqueeze(1)

        # z(alpha) = (1-alpha) z0 + alpha z1.
        path_z = (1 - alpha) * z[0:1] + alpha * z[1:2]

        # Decoding the path tests whether neighbouring latent points change smoothly.
        decoded = model.decode(path_z)

    save_tensor_grid(
        decoded,
        out / "m4_latent_interpolation.png",
        "Latent interpolation: z0 -> z1",
        nrow=9,
        show=not test,
    )
    print("Saved latent interpolation with", len(decoded), "decoded points.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    main(parser.parse_args().test)
```

**Ejecutar:** `python milestone_04_latent_space.py`

**Debes ver:** `Saved latent interpolation with 9 decoded points.` y `outputs/m4_latent_interpolation.png` con 9 imágenes en una fila que transicionan de la imagen 0 a la imagen 1.

**Explicar (checkpoint):** interpolar linealmente es recorrer la recta `z(α) = (1−α)·z0 + α·z1` en el espacio latente y decodificar cada punto; un cambio gradual indica **continuidad local** del decoder, **no** prueba que el espacio esté semánticamente organizado ni “desenredado” (disentangled).

**THINK / CHANGE / TEST** (en `experiments/m4_pairs.py`, sin tocar el oficial)
- *Dos imágenes fuente distintas:* el loader mezcla (`shuffle=True`) pero la semilla es fija, así que el par `x[:2]` siempre es el mismo. Para probar otros pares usa otros índices (`x[2:4]`, `x[4:6]`, …) o selecciona por etiqueta (el loader devuelve `labels`; p. ej. una camiseta y una sandalia vs. dos camisetas). Guarda una figura por par y compara: ¿todos los pares interpolan igual de suave? (Típicamente pares de clases similares dan transiciones más limpias; pares muy distintos muestran “fantasmas” intermedios.)
- *9 → 3 → 20 puntos:* cambia el `9` de `torch.linspace(0, 1, 9, ...)` (y `nrow` acorde). **La trayectoria subyacente es la misma recta** en el espacio latente; solo cambia **cuán densamente se muestrea**.

**✅ Siguiente paso solo si:** la imagen de interpolación existe y puedes explicar por qué no demuestra organización semántica.

---

### Milestone 5 — Experimento controlado del cuello de botella

**Archivo:** `Workshop_1_Autoencoders/milestone_05_bottleneck_experiment.py`

```python
"""Milestone 5 - Controlled experiment: compare several latent dimensions."""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workshop_config import get_config, device
from course_utils import fashion_loader, ensure_output, seed_everything
from milestone_02_model import ConvAutoencoder


def train_once(latent_dim, loader, dev, max_batches):
    """Train one fresh model briefly while changing only latent dimension."""
    model = ConvAutoencoder(latent_dim).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    losses = []

    for batch_index, (x, _) in enumerate(loader):
        if batch_index >= max_batches:
            break

        x = x.to(dev)
        optimizer.zero_grad()
        loss = loss_fn(model(x), x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return sum(losses) / len(losses)


def main(test=False):
    seed_everything()
    cfg = get_config("autoencoder")
    dev = device()
    out, _ = ensure_output(Path(__file__).parent)

    # Keep data, optimiser and training budget fixed so latent dimension is the
    # controlled independent variable in this experiment.
    loader = fashion_loader(
        cfg["batch_size"],
        min(cfg["max_train"], 6000),
        test_mode=test,
    )
    latent_dimensions = [4, 16, 64]
    max_batches = 2 if test else 60
    results = []

    for latent_dim in latent_dimensions:
        mean_loss = train_once(latent_dim, loader, dev, max_batches)
        results.append((latent_dim, mean_loss))
        print("latent", latent_dim, "mean training loss", round(mean_loss, 5))

    # CSV output encourages students to analyse data rather than screenshots only.
    with open(out / "m5_bottleneck_results.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["latent_dim", "mean_training_loss"])
        writer.writerows(results)

    fig, ax = plt.subplots()
    ax.plot(
        [latent for latent, _ in results],
        [loss for _, loss in results],
        marker="o",
    )
    ax.set(
        xlabel="Latent dimension",
        ylabel="Mean MSE",
        title="Controlled bottleneck experiment",
    )
    fig.tight_layout()
    fig.savefig(out / "m5_bottleneck.png", dpi=150)

    if not test:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    main(parser.parse_args().test)
```

**Ejecutar:** `python milestone_05_bottleneck_experiment.py`

**Debes ver:** una línea `latent <d> mean training loss <valor>` para `d = 4, 16, 64`, el CSV `outputs/m5_bottleneck_results.csv` y `outputs/m5_bottleneck.png`.

📝 **Copia la tabla del CSV** en tu reporte. No asumas el resultado: lo esperable es que la pérdida **tienda** a bajar con más dimensiones latentes, pero con solo 60 batches puede haber desviaciones — interprétalo como **indicativo, no definitivo**.

**Explicar (checkpoint):** por qué una comparación controlada cambia **un solo factor** a la vez (aquí `latent_dim`; datos, optimizador, lr, semilla y presupuesto de entrenamiento quedan fijos); por qué una comparación tan corta (≤60 batches, una sola semilla) es indicativa y no un ranking definitivo de modelos.

**THINK / CHANGE / TEST**
- *Añadir `latent_dim = 2`:* **primero escribe tu predicción** en el reporte (p. ej. “pérdida más alta que 4, porque 2 números no alcanzan a codificar forma + textura…”) y luego crea `experiments/m5_with_2.py` con `latent_dimensions = [2, 4, 16, 64]` (guardando en `m5_bottleneck_results_with2.csv/png`). Compara predicción vs. resultado.
- *Denoising autoencoder (DAE) — solo diseño, en 2 líneas conceptuales:* en el bucle de entrenamiento hay que (1) **corromper la entrada** y (2) **calcular la pérdida contra la imagen limpia**. Es decir, el modelo recibe `x_noisy` y el target sigue siendo `x`. Ejemplo conceptual de las dos líneas que cambian (escríbelas con tus palabras/variante propia en el reporte):

  ```python
  x_noisy = (x + 0.3 * torch.randn_like(x)).clamp(0, 1)   # 1) corromper la entrada
  loss = loss_fn(model(x_noisy), x)                        # 2) target = imagen limpia
  ```

  El taller pide no copiar una solución completa de internet; estas dos líneas son la idea mínima — razona por qué `clamp(0,1)` (mantener el rango de píxeles) y qué nivel de ruido elegirías.

**✅ Final de los milestones:** CSV + PNG generados y reporte de predicción vs. resultado escrito.

---

## 6. Preguntas finales del taller (responder con TUS resultados)

El taller prohíbe copiar texto: responde con tus outputs, cambios controlados y razonamiento técnico. Pistas de **qué evidencia usar** (no son las respuestas):

1. **¿Qué información se pierde con un bottleneck muy pequeño y cómo detectarlo en imágenes y no solo con la pérdida?** → Compara visualmente reconstrucciones con `latent_dim = 2, 4, 16, 64` sobre el **mismo batch**: busca pérdida de detalles finos (texturas, logos, cuellos), bordes borrosos, prendas parecidas que se “confunden” entre sí. Usa la figura de M3/M5 y tus corridas con `latent_dim=2`.
2. **¿Por qué un autoencoder puede aprender la identidad si el bottleneck/capacidad es muy generoso?** → Piensa en `latent_dim ≥ dimensión efectiva de la entrada` (784) y capacidad alta: no hay presión de compresión, copiar es un óptimo trivial. Relaciónalo con tus resultados de `latent_dim = 64` frente a 4.
3. **MSE vs. entropía cruzada binaria (BCE) para imágenes normalizadas: ¿qué suposiciones implican?** → MSE ⇒ ruido gaussiano de varianza constante sobre el píxel; BCE ⇒ cada píxel como probabilidad Bernoulli en `[0,1]` (encaja con Sigmoid), penaliza más fuerte los errores en los extremos 0/1 y da gradientes diferentes. Si quieres evidencia empírica, haz una variante en `experiments/` con `nn.BCELoss()` y compara reconstrucciones.
4. **¿Cómo probar si el latente sirve para un clasificador sin cambiar el objetivo del autoencoder?** → Congela el encoder, extrae `z = model.encode(x)` para train/test **con etiquetas**, entrena un clasificador simple (regresión logística / `nn.Linear`) sobre `z` y mide accuracy en test; compara contra un baseline (píxeles crudos y/o `latent_dim` distintos). Es lo que sugiere el diseño del Milestone 3.
5. **¿Qué cambios convierten el modelo en denoising autoencoder?** → Corromper la entrada, mantener el target limpio (las dos líneas del Milestone 5); opcionalmente evaluar con entradas ruidosas nuevas y comparar la reconstrucción con la del modelo estándar.
6. **¿Por qué la interpolación latente es diagnóstico y no prueba formal de organización semántica?** → Usa tus pares del Milestone 4: una transición suave solo demuestra continuidad del decoder en esa línea; no garantiza que ejes/direcciones tengan significado, que todo punto intermedio sea una prenda válida, ni que el espacio esté “desenredado”. Un autoencoder determinista no impone estructura probabilística (a diferencia de un VAE).

---

## 7. Entregable sugerido: `Workshop_1_Autoencoders/REPORTE_TALLER_1.md`

Pide a Claude Code que, al terminar cada milestone, vaya completando este archivo **con datos reales de tus corridas** (y tú redactas/validas las explicaciones):

```markdown
# Reporte Taller 1 — Autoencoders
## Entorno
- Perfil: ... | Dispositivo: ... | PyTorch: ...
- batch_size / epochs / max_train / latent_dim: ...
## M1 — Datos
- Batch shape: ... | Pixel range: ... | Figura: outputs/m1_batch.png
- Explicación ([B,C,H,W] y rango [0,1]): ...
- Think/Change/Test: ...
## M2 — Modelo
- Shapes: input ... latent ... output ... | Parámetros: ...
- Cálculo de tamaños 28→14→7→14→28: ...
- Sigmoid vs ReLU: ...
## M3 — Entrenamiento
- Pérdida por época: ... | Pérdida final (lr=1e-3): ...
- Experimento lr=2e-3: pérdida final ... | observación de estabilidad: ...
- Por qué baja MSE ≠ latente útil para clasificar: ...
## M4 — Espacio latente
- Pares probados: ... | ¿Interpolan igual de suave?: ...
- 3 vs 9 vs 20 puntos: ...
## M5 — Bottleneck
| latent_dim | mean_training_loss |
|---|---|
| 4 | ... |
| 16 | ... |
| 64 | ... |
- Predicción para latent_dim=2: ... | Resultado: ...
- Diseño del denoising autoencoder (2 líneas): ...
## Preguntas finales (1–6)
1. ... 2. ... 3. ... 4. ... 5. ... 6. ...
```

---

## 8. Checklist final

- [ ] Fase 0: `hardware_profile.json` creado y perfil anotado.
- [ ] 5 archivos `milestone_0X_*.py` creados con los nombres exactos y corridos **sin** `--test`.
- [ ] Existen: `m1_batch.png`, `m2_untrained_recon.png`, `m3_loss.png`, `m3_reconstruction.png`, `m4_latent_interpolation.png`, `m5_bottleneck.png`, `m5_bottleneck_results.csv` y `checkpoints/autoencoder.pt`.
- [ ] Los 5 bloques *Explicar* los puedes responder tú sin leer.
- [ ] Los 5 *Think/Change/Test* hechos en `experiments/` (los milestones oficiales quedaron intactos: `git diff`).
- [ ] 6 preguntas finales respondidas con evidencia propia en `REPORTE_TALLER_1.md`.
- [ ] `.gitignore` creado y commit hecho en tu rama de trabajo (**sin** subir `.venv/` ni `datasets/`).

---

### Anexo — Si algo falla

| Síntoma | Causa probable | Solución |
|---|---|---|
| El script “se queda pegado” al final | `plt.show()` esperando que cierres la ventana | `MPLBACKEND=Agg` (ver §3.2) |
| `ModuleNotFoundError: workshop_config` | Ejecutaste un script movido de carpeta | Mantén los milestones en `Workshop_1_Autoencoders/` dentro de la raíz del proyecto |
| `ModuleNotFoundError: milestone_02_model` en un script de `experiments/` | `experiments/` no tiene el workshop en `sys.path` | Añade la carpeta `Workshop_1_Autoencoders` al `sys.path` (ver M3) |
| `RuntimeError: FashionMNIST is not available` | Sin internet en la primera corrida | Conéctate una vez para descargar a `datasets/` |
| `FileNotFoundError: Run Milestone 3 first...` | Falta `checkpoints/autoencoder.pt` | Corre el Milestone 3 oficial |
| Perfil LOW teniendo GPU NVIDIA | PyTorch instalado sin CUDA | Reinstala PyTorch con CUDA y repite `check_computer.py` |
| `py` no se reconoce (Windows) | Python launcher no instalado | Instala Python 3.10/3.11 (según `Setup_Environment.bat`) |
