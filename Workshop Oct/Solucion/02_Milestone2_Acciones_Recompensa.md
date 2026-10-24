# Paso 2 · Milestone 2 — Acciones, inventario, efectivo y recompensa

**Comando:** `START_HERE.bat` → opción **3** (`python -m examples.demo_03_actions_and_reward`)

## Resultado obtenido
```
action                 price      inv       cash          NAV      reward
-------------------------------------------------------------------------------
BUY_MARKET_SMALL     $  99.46      10 $ 98,999.20 $  99,993.77   -0.6280
BUY_MARKET_SMALL     $  99.38      20 $ 98,003.90 $  99,991.54   -0.2332
HOLD                 $  99.76      20 $ 98,003.90 $  99,999.16    0.7522
HOLD                 $  99.74      20 $ 98,003.90 $  99,998.74   -0.0523
SELL_MARKET_SMALL    $  99.48      10 $ 99,000.60 $  99,995.41   -0.3373
HOLD                 $ 100.06      10 $ 99,000.60 $ 100,001.16    0.5697
BUY_LIMIT_SMALL      $ 100.16      10 $ 99,000.60 $ 100,002.19    0.0984
HOLD                 $ 100.11      10 $ 99,000.60 $ 100,001.72   -0.0525
SELL_LIMIT_SMALL     $  99.39       0 $100,002.50 $ 100,002.50    0.0780
HOLD                 $  98.50       0 $100,002.50 $ 100,002.50    0.0000
...
```

## Explicación (contabilidad)
- **NAV = efectivo + inventario × precio medio.** Es el valor del portafolio "a mercado".
- **BUY** baja el efectivo y sube el inventario; **SELL** lo contrario.
- La primera compra de mercado cuesta: compramos 10 al *ask* ($100.08, el demo usa el escenario
  `volatile` con spread de 15 pb) y valoramos al *mid*, así que perdemos de inmediato medio spread
  por acción, más el movimiento del precio.
  Por eso la recompensa del primer paso es negativa (−0.63).
- `BUY_LIMIT_SMALL` (paso 7) **no se ejecutó**: el inventario siguió en 10. En el mercado de práctica
  las órdenes límite se llenan con probabilidad 55 %. En cambio `SELL_LIMIT_SMALL` (paso 9) sí se
  llenó, y vendió al *ask*: ganamos medio spread frente al mid.
- Con inventario 0 la recompensa es 0 aunque el precio caiga de $99.39 a $97.96: **sin posición no hay riesgo**.

Recompensa del paquete:
```python
reward = pnl_change / 1000.0 - 0.0005 * abs(inventory)
```
El primer término es el cambio de NAV (en unidades de $10); el segundo es una pequeña penalización
por tener inventario.

## Respuesta al checkpoint
**Si solo premiamos la ganancia de corto plazo, ¿qué comportamiento riesgoso puede aprender?**
- Apalancarse al máximo (±500 acciones) en la dirección que funcionó en entrenamiento: en un mercado
  con tendencia alcista "comprar siempre" es óptimo… hasta que el mercado cambia. Es exactamente lo
  que veremos que hace el DQN inicial (Paso 6).
- Ignorar la cola de riesgo: una estrategia con mucha volatilidad y media apenas positiva parece buena.
- Terminar el episodio con mucho inventario (el costo de cerrarlo no aparece en la recompensa).

**¿Cómo penalizar drawdown, inventario o rotación sin que el agente se quede siempre en HOLD?**
- Penalizar el **riesgo** y no la **actividad**: usamos una penalización media-varianza
  `0.5·κ·(|inv|·precio·σ)²`, que crece con el cuadrado de la posición *y* con la volatilidad. Una
  posición pequeña casi no se castiga (se puede seguir ganando), una grande en un mercado volátil sí.
- No agregar un costo por operación extra: las órdenes de mercado **ya pagan** el spread en el entorno;
  sumar otro costo empuja al agente a HOLD.
- Cobrar el costo de liquidar el inventario **solo al final** del episodio.
- Mantener una fuente de ganancia "segura" disponible: las órdenes límite ganan medio spread,
  así que reducir riesgo no obliga a dejar de ganar.

Esto es lo que implementamos en `student_work/envs.py` (ver Paso 8).
