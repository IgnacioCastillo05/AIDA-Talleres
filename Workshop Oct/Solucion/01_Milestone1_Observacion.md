# Paso 1 · Milestone 1 — Ver la observación del mercado

**Comando:** `START_HERE.bat` → opción **1** (o `python -m examples.demo_01_observe_market`)

## Resultado obtenido
```
16-DIMENSION MARKET OBSERVATION
----------------------------------------------
return_1              0.0000
return_5              0.0000
return_20             0.0000
volatility_10         0.0000
spread_bps            0.0200
book_imbalance       -0.0922
bid_depth             0.8170
ask_depth             0.9830
inventory             0.0000
cash_ratio            0.5000
nav_return            0.0000
time_progress         0.0000
open_orders           0.0000
price_vs_ma5          0.0000
price_vs_ma20         0.0000
bias                  1.0000

Take action 0 = HOLD
step=1  price=$ 100.05  inventory=   0  reward= 0.0000  NAV=$100,000.00
step=2  price=$  99.96  inventory=   0  reward= 0.0000  NAV=$100,000.00
step=3  price=$  99.96  inventory=   0  reward= 0.0000  NAV=$100,000.00
```

## Explicación
- El agente **no ve un gráfico**: ve un vector de 16 números normalizados (casi todos entre −1 y 1).
  Así una red neuronal pequeña puede procesarlo sin que una variable grande (p. ej. el precio en
  centavos) "aplaste" a las demás.
- En el paso 0 los retornos y la volatilidad valen 0 porque todavía no hay historia de precios.
- `spread_bps = 0.02` → el spread es 4 pb (4/200 = 0.02), el escenario `stable` tiene 4 pb.
- `cash_ratio = 0.5` → tenemos el 100 % del efectivo inicial (se divide por 2 para que quepa en 0..1).
- Con HOLD el inventario es 0, por eso aunque el precio se mueva la NAV no cambia y la recompensa es 0.

## Respuesta al checkpoint
**¿Qué variables describen el mercado externo y cuáles nuestro portafolio?**

| Mercado externo | Estado propio (portafolio) | Contexto |
|---|---|---|
| 0 `return_1`, 1 `return_5`, 2 `return_20` | 8 `inventory` | 11 `time_progress` |
| 3 `volatility_10` | 9 `cash_ratio` | 15 `bias` (constante 1.0) |
| 4 `spread_bps`, 5 `book_imbalance` | 10 `nav_return` | |
| 6 `bid_depth`, 7 `ask_depth` | 12 `open_orders` | |
| 13 `price_vs_ma5`, 14 `price_vs_ma20` | | |

**¿Por qué el agente necesita ambas?** Porque la mejor acción depende de las dos cosas a la vez.
La misma señal de mercado ("el precio sube") implica acciones distintas si ya estoy lleno
(inventario = +500, al límite: no puedo comprar más y mi riesgo es máximo) o si estoy plano.
Sin el estado propio el problema no sería Markoviano: la recompensa futura depende del
inventario y del efectivo, no solo del mercado. Y `time_progress` permite aprender a cerrar
posiciones antes del final.
