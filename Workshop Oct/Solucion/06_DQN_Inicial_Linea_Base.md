# Paso 6 · Entrenar y probar el DQN inicial (línea base)

**Comandos:**
```bat
.venv\Scripts\python.exe -m student_work.train_starter --steps 5000 --seed 7 --out outputs\starter_model.pt
.venv\Scripts\python.exe -m shared.test_policy --model outputs\starter_model.pt --scenario all --seed 101
```
(`train_starter.py` es una copia exacta del `train.py` original del paquete; la opción 8 del menú hace
la misma prueba sobre `outputs\model.pt`.)

## Qué hace el trainer original
- Cicla los escenarios `stable → trending → volatile`, episodios de 250 pasos.
- Exploración ε-greedy: ε baja linealmente de 1.0 a 0.05 durante el 75 % del entrenamiento.
- *Replay buffer* de 50.000 transiciones, lotes de 64.
- Red objetivo copiada cada 250 pasos, γ = 0.995, pérdida Huber, *gradient clipping* a 5.
- Exporta a TorchScript con `torch.jit.script`.

## Resultado (prueba oficial del paquete, una semilla por régimen)
```
stable:   NAV=$98,838.26,  PnL=$-1,161.74, trades=107
trending: NAV=$102,912.71, PnL=$2,912.71,  trades=192
volatile: NAV=$94,717.92,  PnL=$-5,282.08, trades=215
```

## Diagnóstico
- Gana solo en `trending`. Revisando el inventario final en la evaluación del Paso 7 (≈ 420 acciones
  en promedio) se ve que aprendió básicamente **"comprar y mantener"**: en el mercado de práctica la
  única tendencia es alcista, así que eso es lo que más recompensa dio en entrenamiento.
- En `volatile` esa misma posición grande produce pérdidas de miles de dólares: la recompensa original
  casi no penaliza el riesgo (0.0005·|inv| es muy pequeño frente a la volatilidad).
- 5.000 pasos son solo 20 episodios: la mayor parte del tiempo el agente está explorando al azar.

Conclusión: **es un baseline funcional, no una solución.** Sus problemas (sesgo direccional, riesgo
sin control, poca diversidad de entrenamiento) guían las mejoras del Paso 8.
