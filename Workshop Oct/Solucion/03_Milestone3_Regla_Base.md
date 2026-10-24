# Paso 3 · Milestone 3 — Una regla transparente antes de RL

**Comando:** `START_HERE.bat` → opción **2** (`python -m examples.demo_02_rule_based_agent`)

## Resultado obtenido (regla de momentum del demo)
```
stable     final NAV=$99,935.29  PnL=$-64.71  trades=87  inventory=110
trending   final NAV=$101,547.36  PnL=$1,547.36  trades=80  inventory=400
volatile   final NAV=$100,419.19  PnL=$419.19  trades=132  inventory=280
```

## Explicación
La regla compra 10 si el retorno de 5 pasos es > umbral y vende 10 si es < −umbral.
- En `trending` gana porque la tendencia hace que el momentum casi siempre sea positivo: termina
  con 400 acciones y se beneficia de la subida.
- En `stable` pierde: cada operación de mercado paga el spread y el "momentum" es ruido.
- En `volatile` gana con esta semilla… pero termina con 280 acciones de riesgo abierto.

Una sola prueba exitosa **no basta**: el mismo código se comporta muy distinto según el régimen.

## Tarea del estudiante: cambiar UN umbral
Implementado en `student_work/exp_threshold.py`:
```bat
.venv\Scripts\python.exe -m student_work.exp_threshold
```
Resultado:
```
A) Semilla del demo (19), 150 pasos
  umbral               stable             trending             volatile
   0.005        -94 (124 tr)      1,531 ( 94 tr)        494 (142 tr)
   0.015        -65 ( 87 tr)      1,547 ( 80 tr)        419 (132 tr)
   0.030        -46 ( 38 tr)      1,513 ( 66 tr)        472 (123 tr)
   0.060         -6 (  5 tr)      1,362 ( 46 tr)        661 (101 tr)

B) 20 semillas (1000..1019), 300 pasos: PnL medio +/- desviación
  umbral               stable             trending             volatile
   0.005        -87 +/-   179      3,416 +/-   792        159 +/- 1,721
   0.015        -67 +/-   176      3,409 +/-   785        197 +/- 1,804
   0.030        -40 +/-   105      3,370 +/-   801        288 +/- 1,872
   0.060        -11 +/-    33      3,119 +/-   824        216 +/- 1,766
```

### Conclusión registrada
- Subir el umbral **reduce operaciones** y por lo tanto la pérdida por spread en `stable`
  (−87 → −11), pero **sacrifica ganancia** en `trending` (3,416 → 3,119). La mejora no es en todos
  los regímenes: **desplaza** el desempeño de uno a otro.
- Con una sola semilla, en `volatile` el umbral 0.060 parece el mejor (+661). Con 20 semillas las
  diferencias entre umbrales (~±$100) son **mucho menores que la desviación entre semillas (~$1,800)**:
  es ruido, no mejora. Esta es la lección central del taller sobre generalización.

## Otras reglas de referencia (para comparar al RL)
En `student_work/baselines.py` dejamos tres políticas sin aprendizaje:

| Política | Idea |
|---|---|
| `hold` | nunca opera (PnL = 0, el piso de "no hacer daño") |
| `momentum` | la regla del demo |
| `market_maker` | cotiza órdenes límite grandes siempre del lado que reduce el inventario: gana medio spread por ejecución con riesgo bajo |

Las evaluamos en el Paso 7 con 6 regímenes × 20 semillas.
