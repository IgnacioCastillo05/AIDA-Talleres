# Paso 4 · Milestone 4 — La política como red neuronal

**Comando:** `START_HERE.bat` → opción **4** (`python -m examples.demo_04_network_forward`)

## Resultado obtenido
```
Input tensor shape : (1, 16)
Output tensor shape: (1, 9)

Q-values from an UNTRAINED network:
0: HOLD                    0.17781
1: BUY_MARKET_SMALL        0.11251
2: SELL_MARKET_SMALL       0.09426
3: BUY_MARKET_LARGE        0.05012
4: SELL_MARKET_LARGE      -0.06544
5: BUY_LIMIT_SMALL         0.04180
6: SELL_LIMIT_SMALL        0.06995
7: BUY_LIMIT_LARGE        -0.08198
8: SELL_LIMIT_LARGE        0.04308

Argmax action: 0 HOLD
```

## Explicación
- `unsqueeze(0)` convierte el vector `[16]` en un lote `[1, 16]`: las redes de PyTorch siempre
  trabajan por lotes, y el contrato del torneo es `[batch, 16] → [batch, 9]`.
- Cada una de las 9 salidas es un **valor Q**: la estimación de la recompensa futura acumulada si
  tomo esa acción ahora y después sigo la política. La acción elegida es el `argmax`.
- Con pesos aleatorios los valores son casi iguales y la elección (HOLD) es pura casualidad.
  Entrenar = ajustar los pesos para que las acciones útiles tengan valores Q más altos
  en los estados donde convienen.

## Lo que cambiamos en la solución
En `student_work/model.py` mantuvimos `QNetwork` (los demos 4 y 5 lo usan) y agregamos
`DuelingQNetwork`:

```
           ┌─> V(s)    (1 valor: ¿qué tan bueno es este estado?)
16 → 128 → 128
           └─> A(s,a)  (9 valores: ¿qué tan mejor es cada acción?)

Q(s,a) = V(s) + A(s,a) − promedio(A)
```
Por qué: en trading, la mayor parte del valor de un estado depende del inventario y del régimen,
no de la acción puntual. La arquitectura *dueling* aprende `V(s)` con todas las transiciones,
aunque una acción concreta se haya probado poco. El contrato no cambia (entra 16, salen 9).
