# Paso 10 · Respuestas a las preguntas de análisis (sección 18 del documento)

### 1. ¿Por qué reproducir un CSV histórico de precios no equivale a interactuar con un exchange simulado?
Un CSV es **pasivo**: el precio siguiente es el mismo hagamos lo que hagamos. En un exchange simulado
nuestras órdenes **consumen liquidez, mueven el libro y cambian el precio** (impacto de mercado), las
órdenes límite pueden no ejecutarse, y otros agentes reaccionan a nosotros. El estado siguiente depende
de nuestra acción: eso es un problema de decisión secuencial (RL), no de predicción. Un backtest sobre
CSV sobreestima sistemáticamente las estrategias que operan mucho o en tamaño grande, porque asume
ejecución perfecta y sin impacto.

### 2. ¿Qué información del vector de observación ayuda a detectar baja liquidez?
- `spread_bps` (índice 4): spreads anchos son la señal principal.
- `bid_depth` y `ask_depth` (6 y 7): poca profundidad en el libro.
- `book_imbalance` (5): desbalances extremos aparecen cuando un lado del libro está vacío.
- `volatility_10` (3) y saltos en `return_1` (0): con poca liquidez el precio salta más.
- `open_orders` (12): si nuestras órdenes límite se acumulan sin ejecutarse.

### 3. ¿Cuándo es preferible una orden límite a una de mercado?
Cuando **no hay urgencia** y el spread es relevante: la orden límite *gana* medio spread (compra en el
bid / vende en el ask) en vez de *pagarlo*, a cambio de **incertidumbre de ejecución** (en la práctica
se llena el 55 % de las veces) y del riesgo de selección adversa (se nos llena justo cuando el precio
va en contra). Es preferible en spreads anchos / baja liquidez, para hacer *market making* y para
reducir inventario sin pagar costos. La orden de mercado es preferible cuando la señal es fuerte y de
corto plazo, o cuando hay que salir ya (riesgo, crisis, fin de sesión).
Nuestro análisis muestra que en el escenario `low_liquidity` (spread 40 pb) la diferencia es enorme:
la regla con límites gana ~$1,660 de media, la de mercado pierde ~$580.

### 4. ¿Cómo puede una recompensa aparentemente rentable producir una política insegura?
Si la recompensa es solo el PnL, el agente maximiza el **valor esperado** sin importar la varianza:
aprende a usar la posición máxima en la dirección que funcionó en entrenamiento. Es lo que hace el DQN
inicial: en las semillas de práctica (`trending` siempre sube) aprende "comprar y mantener 400–500
acciones", gana mucho en `trending` y pierde fuerte en `volatile`, `downtrend` y `stress`. Otros
riesgos: terminar el episodio con inventario sin pagar el costo de cerrarlo, sobreoperar si la
recompensa no refleja el spread, o explotar un defecto del simulador (p. ej. una probabilidad de
ejecución irreal) que no existe en ABIDES.

### 5. ¿Cómo distinguir generalización real de suerte en una semilla?
- Evaluar en **muchas semillas** (usamos 20 por régimen) y reportar media, desviación, peor caso y
  % de semillas ganadoras, no solo la mejor corrida.
- Comparar la diferencia entre modelos con la **dispersión entre semillas** (ver Paso 3: en
  `volatile` la diferencia entre umbrales era ~$100 y la desviación ~$1,800 → ruido).
- **Separar semillas**: entrenamiento (< 5000), validación para escoger el checkpoint (5000+) y test
  final (9000+) que no se mira hasta el final. Si se escoge el modelo mirando repetidamente el test,
  el test se convierte en entrenamiento.
- Evaluar en **regímenes no idénticos** a los de entrenamiento y repetir el entrenamiento con otra
  semilla de inicialización.

### 6. ¿Por qué una política puede comportarse distinto cuando varios estudiantes operan a la vez?
Porque los otros agentes **son parte del entorno** y no son estacionarios: consumen la liquidez que
nuestra política esperaba, ensanchan el spread, mueven el precio con sus órdenes (impacto), compiten
por las mismas ejecuciones límite (prioridad precio-tiempo) y pueden tener estrategias parecidas
(si todos compran momentum, la señal se autoalimenta y luego colapsa). La política fue entrenada en
un mercado donde solo ella actuaba: en el torneo cambia la dinámica de transición, que es justamente
el componente multi-agente (MARL).

### 7. ¿Qué compromiso hicimos entre ganancia, drawdown y actividad?
Escogimos la política por un **score robusto = PnL medio − ½·drawdown medio**, promediado entre seis
regímenes, en vez de por la ganancia máxima. Aceptamos ganar menos que el DQN inicial en `trending`
a cambio de no perder en los mercados bajistas, volátiles y de crisis, y de terminar con poco
inventario. La actividad (número de operaciones) es alta pero casi toda con órdenes límite, que
**ganan** el spread en lugar de pagarlo. Ver cifras en el Paso 8.

### 8. Si la política falla en una crisis, ¿qué evidencia distingue entre estado, recompensa, distribución de entrenamiento o algoritmo?
| Hipótesis | Evidencia que la confirmaría |
|---|---|
| **Representación del estado** | La política toma la misma acción antes y durante el choque; las observaciones de crisis y normales son casi idénticas (p. ej. `volatility_10` saturada en 1.0); ablaciones de features no cambian nada; una regla con información extra (p. ej. volatilidad más larga) sí reacciona. |
| **Recompensa** | La política hace exactamente lo que la recompensa premia (los valores Q de la acción mala son altos) y el PnL de entrenamiento era bueno; al subir la aversión al riesgo el comportamiento cambia. |
| **Distribución de entrenamiento** | Desempeño bueno en regímenes vistos y malo solo en los no vistos; al agregar escenarios de estrés al entrenamiento mejora. |
| **Algoritmo / optimización** | La curva de validación es inestable, Q diverge o sobreestima mucho el retorno real, resultados muy distintos entre semillas de entrenamiento; Double DQN, más pasos o menor tasa de aprendizaje lo corrigen. |

En la solución hicimos ese análisis con **ablaciones** (Paso 8): apagamos una mejora a la vez para ver
cuál explica la diferencia.
