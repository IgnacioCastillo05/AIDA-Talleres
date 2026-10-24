# Paso 11 · Bitácora experimental (evidencia mínima, sección 17 del documento)

| Requisito | Nuestra evidencia |
|---|---|
| **Algoritmo y arquitectura** | Double DQN con retornos de 3 pasos y red objetivo suave (τ = 0.005). Red *dueling*: 16 → 128 → 128 → {V: 64 → 1, A: 64 → 9}. Exportada como TorchScript (`torch.jit.script`). |
| **Función de recompensa** | `dNAV/1000 − 0.5·κ·(|inv|·precio·σ/1000)²`, κ = 0.05, más el costo de liquidar el inventario (medio spread por acción) al final del episodio. `student_work/envs.py`. |
| **Escenarios y semillas de entrenamiento** | 50 % regímenes con nombre (stable, trending, volatile, downtrend, low_liquidity, stress) y 50 % aleatorios. Semillas de entorno `7 + episodio` (< 5000). Semilla global 7. |
| **Hiperparámetros** | 150.000 pasos, γ = 0.99, n = 3, lr = 3e-4 (Adam), lote 128, replay 100.000, warmup 2.000, ε 1.0 → 0.02 en el 50 % de los pasos, clipping 10, Huber. |
| **Curva de entrenamiento** | `Student_Package/outputs/training_curve.png` y `outputs/model.csv` (validación cada 5.000 pasos). |
| **Desempeño por escenario y semilla** | `outputs/eval_test_final.md`: 6 regímenes × 20 semillas de test (9000+), comparado con hold, momentum, market maker y el DQN inicial. |
| **Por qué se eligió el modelo final** | (1) mejor checkpoint en validación (paso 110.000), no el último; (2) κ = 0.05 por el compromiso entre ganancia y drawdown (Paso 8); (3) mejor score robusto en test que todas las referencias (1,360 vs 707 / 547 / 0 / −256). |
| **Una limitación o falla observada** | `stable` es el régimen más débil (50 % de semillas ganadoras): a veces confunde ruido con tendencia. Además, la ganancia por capturar el spread depende de que las órdenes límite se llenen el 55 % de las veces, algo que puede no cumplirse en ABIDES ni en el torneo. |

## Registro cronológico de experimentos
| # | Experimento | Resultado clave | Decisión |
|---|---|---|---|
| 1 | Demos 1–5 | Todo corre; pérdida finita en la actualización DQN | Continuar |
| 2 | Barrido del umbral de momentum (4 umbrales × 20 semillas) | Las diferencias en `volatile` son menores que el ruido | Evaluar siempre con muchas semillas |
| 3 | DQN inicial, 5k pasos | Score −256; solo gana en `trending` (compra y mantiene) | Diagnóstico: sesgo direccional y riesgo sin controlar |
| 4 | Reglas base en 6 regímenes | Market maker: 547 con drawdown mínimo; momentum: 707 con drawdowns altos | Meta: superar a ambos |
| 5 | Prueba corta del trainer nuevo (20k pasos) | Aprende pero con ruido | Entrenar 150k y guardar el mejor checkpoint |
| 6 | κ ∈ {0.01, 0.05, 0.2} | 1,729 / 1,567 / 930 en validación; drawdown 1,200 / 410 / 180 | κ = 0.05 |
| 7 | Evaluación final en test | 1,360; positivo en los 6 regímenes | Modelo final = `outputs/model.pt` |
| 8 | 5 ablaciones + 1 semilla extra | Regímenes aleatorios y n-step: claves. Double/Dueling: sin efecto medible | Documentado en el Paso 8 |
| 9 | Exportación de `TEAM_XX.zip` | `VALID` | Re-exportar con el ID real del equipo |

## Archivos de resultados (`Student_Package/outputs/`)
| Archivo | Contenido |
|---|---|
| `model.pt` / `model.csv` | Modelo final y su curva de validación |
| `training_curve.png` | Gráfica de la curva |
| `eval_test_final.md` | Tabla final en test (modelo, DQN inicial y reglas) |
| `starter_model.pt` | DQN inicial (línea base) |
| `exp_ra*.pt/.csv` | Experimentos de aversión al riesgo |
| `abl_*.pt/.csv`, `eval_abl_*.md` | Ablaciones y sus evaluaciones |
