# Paso 5 · Milestone 5 — Una actualización DQN real

**Comando:** `START_HERE.bat` → opción **5** (`python -m examples.demo_05_one_dqn_update`)

## Resultado obtenido
```
Transitions collected: 64
One DQN update completed. Loss = 0.106098
```
Una pérdida finita (no NaN/Inf) confirma que el mecanismo funciona.

## Explicación paso a paso
1. Se juntan 64 transiciones `(estado, acción, recompensa, siguiente_estado, terminó)` con acciones aleatorias.
2. **Predicción:** `q(states).gather(1, actions)` → el valor Q que la red da a la acción que de verdad se tomó.
3. **Objetivo (ecuación de Bellman):**
   ```
   Q_target = r + γ · (1 − done) · max_a Q_target(s', a)
   ```
   "Lo que gané ahora + lo mejor que creo que puedo ganar desde el siguiente estado".
   `(1 − done)` corta el futuro cuando el episodio termina.
4. **Pérdida Huber** (`smooth_l1_loss`): como el MSE para errores pequeños, pero lineal para errores
   grandes; con recompensas de trading (colas pesadas, saltos) evita gradientes explosivos.
5. `backward()` + `optimizer.step()` mueven los pesos para acercar la predicción al objetivo.

Una actualización no es entrenamiento: el trainer repite esto miles de veces con *replay*,
exploración y red objetivo.

## Respuesta al checkpoint
**¿Por qué se usa una red objetivo copiada en vez de la misma red cambiante en ambos lados?**

Porque si el objetivo se calcula con la misma red que se está actualizando, **cada paso de
gradiente también mueve el objetivo**: se persigue un blanco que se mueve con uno mismo. Eso produce
bucles de retroalimentación (sobreestimar Q en `s'` sube el objetivo de `s`, que a su vez sube Q en
`s'`…), oscilaciones y divergencia. Congelar una copia (la red *target*) y actualizarla lentamente
convierte cada tramo del entrenamiento en un problema de regresión casi supervisado con objetivos
estables.

En la solución usamos dos refinamientos de esta idea:
- **Actualización suave (Polyak, τ = 0.005):** en vez de copiar toda la red cada 250 pasos,
  `target ← 0.995·target + 0.005·online` en cada paso. El objetivo cambia de forma continua y lenta.
- **Double DQN:** la red online **elige** la mejor acción en `s'` y la red target la **evalúa**.
  El `max` de valores ruidosos está sesgado hacia arriba (siempre escoge el ruido más optimista);
  separar elección y evaluación reduce esa sobreestimación, algo importante con recompensas tan
  ruidosas como el PnL.
