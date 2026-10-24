# Paso 9 · Exportar y validar el entregable del torneo

## 1. Dejar el modelo final en `outputs/model.pt`
`python -m student_work.train` ya guarda allí el **mejor checkpoint según validación** (TorchScript).
Si quieres reentrenar exactamente la configuración final, ver el comando en el Paso 8.

## 2. Exportar
`START_HERE.bat` → opción **9**, o directamente:
```bat
.venv\Scripts\python.exe -m shared.export_submission --team-id TEAM_05 --team-name "Mi equipo" ^
    --members "Nombre 1, Nombre 2" --algorithm "Double Dueling DQN (n-step, risk-adjusted reward)" ^
    --model outputs\model.pt
```
Reemplaza `TEAM_05` por el identificador real del equipo. El script:
1. carga el modelo con `torch.jit.load` en CPU (como el profesor);
2. comprueba que una entrada `[1,16]` produce `[1,9]`;
3. escribe `submission\TEAM_05.zip` con `model.pt` y `metadata.json` **en la raíz**;
4. lo valida y lo borra si no pasa.

## 3. Validar el ZIP exacto que se va a subir
`START_HERE.bat` → opción **V**, o:
```bat
.venv\Scripts\python.exe -m shared.validate_submission submission\TEAM_05.zip
```
Salida esperada:
```
VALID: submission\TEAM_05.zip
```

## Ejemplo ya generado
En `Student_Package/submission/TEAM_XX.zip` dejamos un ZIP de ejemplo generado y validado con el modelo
final (solo con nombres de ejemplo). **Vuelve a exportarlo con el ID y los integrantes reales.**

`metadata.json` generado:
```json
{
  "team_id": "TEAM_XX",
  "team_name": "Equipo Ejemplo",
  "members": ["Integrante 1", "Integrante 2"],
  "model_format": "torchscript",
  "contract_version": "1.0",
  "observation_size": 16,
  "action_size": 9,
  "training_algorithm": "Double Dueling DQN (n-step, risk-adjusted reward)",
  "notes": "..."
}
```

## Errores típicos
| Error del validador | Causa |
|---|---|
| `ZIP root must contain exactly [...]` | Los archivos quedaron dentro de una carpeta, o hay archivos extra. |
| `Cannot load/run model.pt as TorchScript` | Se guardó con `torch.save(state_dict)` en vez de `torch.jit.script(...).save(...)`. |
| `model output shape must be (1, 9)` | Se cambió la capa de salida. |
| `contract_version must be 1.0` | Se modificó `shared/contract.py`. |

## Checklist final (sección 20 del documento)
- [x] Los 5 mini-ejemplos corren.
- [ ] Prueba de humo de ABIDES-NG (opción 6) — correr en el equipo Windows tras `SETUP_ONCE.bat`.
- [x] El modelo mejorado entrena sin NaN/Inf.
- [x] Se probó en 6 regímenes y 20 semillas por régimen.
- [x] `outputs/model.pt` carga en CPU.
- [ ] ZIP final con el `TEAM_XX` real.
- [x] La raíz del ZIP tiene exactamente `model.pt` y `metadata.json`.
- [ ] El validador imprime `VALID` para el ZIP exacto que se sube.
