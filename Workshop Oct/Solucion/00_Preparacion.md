# Paso 0 · Preparación del entorno

## Qué pide el taller
1. Leer `MARL_Stock_Exchange_Workshop.docx`.
2. Ejecutar `SETUP_ONCE.bat` una sola vez: crea `.venv` e instala NumPy, PyTorch, Matplotlib y ABIDES-NG.
3. Ejecutar `START_HERE.bat` y recorrer las opciones 1 a 5 antes de entrenar.

## Cómo hacerlo en Windows (VS Code)
```bat
cd "Workshop Oct\Student_Package"
SETUP_ONCE.bat
START_HERE.bat
```
Si prefieres la terminal de VS Code en lugar del menú:
```bat
.venv\Scripts\python.exe -m examples.demo_01_observe_market
```
> Siempre se ejecuta **desde la raíz de `Student_Package`** y con `-m paquete.modulo`,
> porque los scripts importan `shared.*` y `student_work.*` como paquetes.

## Las reglas del juego (lo que NO se puede tocar)
| Carpeta | ¿Se puede editar? | Por qué |
|---|---|---|
| `shared/` | **No** | Es el contrato común: el profesor usa exactamente la misma observación (16 valores) y las mismas acciones (9). |
| `examples/` | No hace falta | Son demos de aprendizaje. |
| `student_work/` | **Sí** | Aquí va toda nuestra solución. |

El modelo final debe ser **TorchScript**, recibir `float32 [batch, 16]` y devolver `float32 [batch, 9]`.
El torneo toma `argmax(salida)` como la acción.

## Nota sobre ABIDES-NG (opción 6)
ABIDES-NG es el simulador oficial que usa el profesor para la evaluación. La opción 6
solo hace una "prueba de humo" (que esté instalado y corra un día de mercado estable).
No se usa para entrenar: entrenamos en el mercado de práctica (`shared/practice_market.py`),
que es rápido y respeta el mismo contrato. Ejecútala en tu equipo después del setup:
```bat
.venv\Scripts\python.exe -m shared.abides_smoke_test --scenario stable_day --seed 42
```
