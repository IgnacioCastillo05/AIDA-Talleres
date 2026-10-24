# Paso 7 · Construir una evaluación robusta (antes de mejorar nada)

Antes de cambiar el modelo hace falta una forma **confiable** de saber si un cambio mejora. La prueba del
paquete (`shared/test_policy.py`) usa una sola semilla por régimen: vimos en el Paso 3 que eso es
mayormente ruido.

## Qué implementamos (`student_work/evaluate.py`)
- **6 regímenes**: los 3 de práctica (`stable`, `trending`, `volatile`) y 3 extra definidos en
  `student_work/envs.py` que imitan lo que el documento anuncia para la evaluación:

  | Régimen | drift/paso | volatilidad | spread | profundidad | choques |
  |---|---|---|---|---|---|
  | `downtrend` | −0.035 % | 0.12 % | 6 pb | 750 | no |
  | `low_liquidity` | 0 | 0.15 % | 40 pb | 150 | no |
  | `stress` | −0.05 % | 0.30 % | 25 pb | 300 | 5 % de prob., ±4 % |

- **20 semillas por régimen** y 300 pasos por episodio (igual que la prueba oficial).
- **Métricas**: PnL medio, desviación, peor caso, % de semillas ganadoras, *drawdown* máximo medio,
  número de operaciones e inventario final.
- **Un solo número para comparar modelos (score robusto):**
  `promedio entre regímenes de (PnL medio − 0.5 × drawdown medio)`. Premia ganar sin grandes caídas.
- **Semillas separadas**: entrenamiento < 5000, validación 5000+ (escoger checkpoint), test 9000+
  (solo para el reporte final).

Las políticas se cargan **exactamente como lo hará el profesor**: `torch.jit.load(..., map_location="cpu")`
y `argmax` de la salida.

## Comando
```bat
.venv\Scripts\python.exe -m student_work.evaluate --model outputs\starter_model.pt --baselines hold momentum market_maker
```

## Resultado (semillas de test, 20 por régimen)
| política | escenario | PnL medio $ | desv. $ | peor $ | % semillas ganadoras | drawdown medio $ | trades | inventario final abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| momentum | stable | -4 | 232 | -234 | 30% | 262 | 183 | 226 |
| momentum | trending | 3,123 | 828 | 1,149 | 100% | 292 | 146 | 385 |
| momentum | volatile | 235 | 2,491 | -2,216 | 50% | 2,406 | 266 | 217 |
| momentum | downtrend | 3,238 | 690 | 1,951 | 100% | 261 | 138 | 390 |
| momentum | low_liquidity | -578 | 383 | -997 | 10% | 831 | 240 | 194 |
| momentum | stress | 2,126 | 2,956 | -2,985 | 75% | 3,742 | 244 | 312 |
| **momentum** | **score robusto** | **707** | | | | | | |
| market_maker | stable | 179 | 39 | 94 | 100% | 20 | 166 | 20 |
| market_maker | trending | 529 | 83 | 344 | 100% | 22 | 165 | 15 |
| market_maker | volatile | 691 | 354 | -82 | 95% | 243 | 164 | 30 |
| market_maker | downtrend | -10 | 54 | -140 | 55% | 86 | 163 | 30 |
| market_maker | low_liquidity | 1,661 | 173 | 1,318 | 100% | 28 | 164 | 25 |
| market_maker | stress | 649 | 728 | -359 | 80% | 442 | 165 | 22 |
| **market_maker** | **score robusto** | **547** | | | | | | |
| starter_model | stable | 32 | 625 | -1,303 | 60% | 632 | 142 | 418 |
| starter_model | trending | 4,334 | 1,015 | 3,254 | 100% | 373 | 172 | 421 |
| starter_model | volatile | -732 | 1,732 | -4,296 | 40% | 3,193 | 205 | 190 |
| starter_model | downtrend | -260 | 916 | -2,678 | 55% | 1,377 | 200 | 318 |
| starter_model | low_liquidity | -119 | 292 | -790 | 35% | 497 | 200 | 211 |
| starter_model | stress | -56 | 2,463 | -5,134 | 55% | 3,396 | 195 | 206 |
| **starter_model** | **score robusto** | **-256** | | | | | | |

(`hold` da 0 en todo: es el piso.)

## Lectura
1. **El DQN inicial es peor que no hacer nada** en score robusto (−256 vs 0). Su ganancia en
   `trending` es real, pero viene de una apuesta direccional que no generaliza.
2. **Momentum** sigue tendencias en ambas direcciones (por eso gana en `downtrend`), pero con
   drawdowns enormes en `volatile` y `stress`, y pierde en baja liquidez porque paga el spread.
3. **Market maker** gana poco pero casi siempre (95–100 % de semillas en la mayoría de regímenes) y con
   drawdowns mínimos: la ventaja estructural del mercado de práctica está en **capturar el spread con
   órdenes límite**, no en predecir el precio (los retornos son un paseo aleatorio).

Esto define la meta del RL: **combinar** la captura de spread del market maker (bajo riesgo) con la
capacidad de aprovechar tendencias del momentum, superando a ambos en score robusto.
