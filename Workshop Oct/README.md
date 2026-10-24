# Workshop Oct — MARL Stock Exchange (solución paso a paso)

Solución del taller **"MARL Stock Exchange Workshop: Build, train, test and export a standardised
trading agent"**. Un agente de *reinforcement learning* recibe 16 variables del mercado y elige una
de 9 órdenes (comprar o vender, a mercado o límite, en tamaño pequeño o grande). El entregable es un
`TEAM_XX.zip` con un modelo TorchScript (`model.pt`) y `metadata.json`.

## Estructura
```
Workshop Oct/
├── README.md                       ← este archivo
├── Solucion/                       ← EXPLICACIÓN paso a paso (leer en orden)
│   ├── 00_Preparacion.md
│   ├── 01_Milestone1_Observacion.md
│   ├── 02_Milestone2_Acciones_Recompensa.md
│   ├── 03_Milestone3_Regla_Base.md
│   ├── 04_Milestone4_Red_Neuronal.md
│   ├── 05_Milestone5_Actualizacion_DQN.md
│   ├── 06_DQN_Inicial_Linea_Base.md
│   ├── 07_Evaluacion_Robusta.md
│   ├── 08_Mejoras_y_Ablaciones.md
│   ├── 09_Exportar_y_Validar.md
│   ├── 10_Preguntas_de_Discusion.md
│   └── 11_Bitacora_Experimental.md
└── Student_Package/                ← paquete original del taller + nuestra solución
    ├── shared/                     ← contrato común (SIN CAMBIOS)
    ├── examples/                   ← mini-ejemplos (SIN CAMBIOS)
    ├── student_work/               ← NUESTRA SOLUCIÓN
    │   ├── model.py                ← + DuelingQNetwork
    │   ├── train.py                ← trainer mejorado (Double/Dueling/n-step/riesgo/validación)
    │   ├── train_starter.py        ← trainer original, para comparar
    │   ├── envs.py                 ← regímenes extra + recompensa ajustada al riesgo
    │   ├── baselines.py            ← reglas de referencia (hold, momentum, market maker)
    │   ├── evaluate.py             ← evaluación 6 regímenes × N semillas
    │   ├── exp_threshold.py        ← experimento del Milestone 3
    │   └── plot_training.py        ← gráfica de la curva de validación
    ├── outputs/                    ← modelos entrenados, curvas y tablas de resultados
    └── submission/TEAM_XX.zip      ← ZIP de ejemplo, validado (re-exportar con el ID real)
```

## Ruta sugerida (para ir resolviendo poco a poco)
| Paso | Qué se hace | Comando (desde `Student_Package`) |
|---|---|---|
| 0 | Instalar el entorno | `SETUP_ONCE.bat` |
| 1–5 | Mini-ejemplos y checkpoints | `START_HERE.bat` opciones 1–5 |
| 3b | Experimento de umbral | `.venv\Scripts\python.exe -m student_work.exp_threshold` |
| 6 | Entrenar y probar el DQN inicial | `.venv\Scripts\python.exe -m student_work.train_starter --out outputs\starter_model.pt` |
| 7 | Evaluación robusta y reglas base | `.venv\Scripts\python.exe -m student_work.evaluate --model outputs\starter_model.pt --baselines hold momentum market_maker` |
| 8 | Entrenar el modelo mejorado | `.venv\Scripts\python.exe -m student_work.train` |
| 9 | Exportar y validar | `START_HERE.bat` opciones 9 y V |
| 10 | Preguntas de discusión | leer `Solucion/10_...` |

> `START_HERE.bat` opción 7 ejecuta `student_work.train --steps 5000 --seed 7`: con nuestro trainer
> eso es solo una corrida corta de prueba. El modelo final se entrena con los valores por defecto
> (ver Paso 8).

RESULTADOS_PLACEHOLDER
