# Evaluador de ejercicios (lagartijas, sentadillas, curl de bíceps)

Práctica de Machine Learning (ESCOM, IPN). Con la webcam, el sistema ve a
la persona haciendo uno de 3 ejercicios y dice, repetición por repetición,
si la hizo **bien** o **qué error tiene** (según el póster del equipo:
cadera caída, no bajar lo suficiente, codo que se adelanta, etc.).

👉 **¿Eres del equipo y vas a grabar tus videos o a entrenar? Ve directo a
[GUIA_EQUIPO.md](GUIA_EQUIPO.md).**

## Idea general

1. **MediaPipe Pose** detecta 33 puntos del cuerpo en cada frame (webcam o
   video). De ahí solo usamos hombro, codo, muñeca, cadera, rodilla,
   tobillo, talón y punta del pie — los mismos que marca el póster.
2. Con esos puntos se calculan **ángulos** (codo, rodilla, inclinación del
   tronco respecto a la vertical, alineación cadera-hombro-tobillo) y un par
   de **señales con signo** (desviación de la cadera hacia arriba/abajo de
   la línea hombro-tobillo, desplazamiento horizontal del codo) — sin esas
   últimas no se puede distinguir, por ejemplo, cadera caída de cadera
   elevada: un ángulo sin signo se ve igual en los dos casos.
3. Un **ángulo clave** por ejercicio (codo en lagartijas/curl, rodilla en
   sentadillas) sirve para **segmentar repeticiones**: el ángulo baja a un
   "valle" (flexión) y vuelve a subir (extensión) → eso es una repetición.
4. De cada repetición se sacan dos cosas: un **vector de características
   agregadas** (mínimo, máximo, media y rango de cada señal) y la
   **secuencia** de cada señal remuestreada a 20 puntos (la forma del
   movimiento, independiente de qué tan rápido se hizo). Un **ensamble por
   ejercicio** (PyTorch) clasifica la repetición en `correcto` o uno de los
   errores del póster:
   - una **CNN-1D híbrida** que convoluciona sobre las secuencias y concatena
     las agregadas antes de la capa final, y
   - un **MLP** sobre las agregadas;
   se promedian sus probabilidades.

### Versión 2 de las características (qué se corrigió y por qué)

Con los datos de las 5 personas el primer modelo (MLP sobre agregadas v1)
no pasaba de F1 ≈ 0.5 en lagartijas y sentadillas. Al revisar las señales
se encontraron errores de diseño, no falta de datos:

| Problema en v1 | Corrección en v2 |
|---|---|
| El signo de la cadera caída/elevada dependía de hacia qué lado miraba la persona en el video. | `desviacion_perpendicular_normalizada` ordena el segmento hombro→tobillo de izquierda a derecha: el signo ya no depende de la orientación. |
| El desplazamiento del codo (curl) también dependía del lado. | `codo_x_norm` se multiplica por la dirección en la que mira la persona (nariz vs. hombro) y se normaliza por el largo del torso. |
| "Talón levantado" se medía con una línea base del primer frame, que no existe si la serie empieza ya abajo. | `elevacion_talon` = (pie.y − talón.y) / largo de la espinilla: se compara el talón con la punta del pie en el mismo frame, sin línea base. |
| La duración en frames dependía de los FPS del celular de cada quien. | Se quitó; la secuencia remuestreada captura el ritmo sin depender de los FPS. |
| Videos duplicados con otro nombre inflaban clases. | `entrenar` quita repeticiones idénticas. |

Cada fila del CSV lleva `version`; `entrenar` ignora las filas de una
versión anterior y avisa quién tiene que volver a correr `extraer`.

### Resultados (validación cruzada por serie)

Con los videos de Gadiel (antes de que el resto del equipo reextraiga con
v2), mismas series, misma partición:

| Ejercicio | v1 (MLP agregadas) | v2 (ensamble CNN-1D + MLP) |
|---|---|---|
| Lagartijas | 0.52 | **1.00** |
| Sentadillas | 0.51 | **0.80** |
| Curl de bíceps | 0.71 | **0.87** |

Modelos comparados sobre v2 (F1 macro): regresión logística, Random
Forest, Gradient Boosting, MLP, CNN-1D y el ensamble. El ensamble fue el
único que quedó entre los mejores en los 3 ejercicios (la CNN-1D sola gana
en lagartijas/sentadillas, el MLP solo en curl). Estas tablas se
actualizarán cuando el equipo completo reextraiga.

### Por qué así y no de otra forma

- **Señales de pose por repetición, no píxeles**: el póster ya redujo el
  problema a ángulos de articulaciones, que es justo lo que hace manejable
  entrenar con pocos videos del equipo (una CNN/LSTM sobre píxeles
  necesitaría muchísimos más datos). La CNN-1D trabaja sobre esas señales
  (≈ 5 canales × 20 puntos), no sobre video, por eso entrena en segundos.
- **Un modelo por ejercicio, no uno solo para los 3**: cada ejercicio
  tiene su propio conjunto de errores y de ángulos relevantes; mezclarlos en
  una sola salida solo complica el problema sin ninguna ventaja.
- **Segmentación por repetición, no clasificación frame a frame**: así es
  como lo pide la práctica ("si cada repetición está bien hecha"), y además
  evita que un solo frame ruidoso (MediaPipe pierde un punto un instante)
  arruine la predicción — se agregan varios frames antes de decidir.

## Decisiones y limitaciones (léelas antes de grabar)

- **Vista lateral, cuerpo completo en cuadro**, igual que el póster. Todos
  los ángulos están pensados para esa cámara; una toma frontal da ángulos
  sin sentido.
- **"Rodillas hacia adentro" (sentadillas) se quedó fuera de esta primera
  versión.** El póster mismo lo marca como "requiere toma frontal": el valgo
  de rodilla no se puede medir desde la vista lateral que usan los demás
  ángulos de sentadillas, y pedir una segunda cámara (o cambiar de ángulo a
  media sesión) complica demasiado la grabación para lo que da. Si el equipo
  quiere agregarlo después, la señal (distancia horizontal rodilla-tobillo
  en vista frontal) se calcularía aparte, con su propio modelo.
- **Lagartijas es el ejercicio más difícil para MediaPipe**: el cuerpo
  horizontal es una pose menos común en los datos con los que se entrenó
  Pose Landmarker, así que pierde puntos más seguido que en los otros dos
  ejercicios (de pie). Ayuda grabar un poco en diagonal (no al ras del
  piso) y con buena luz. Es justo lo que ya anotó el póster con lo de
  "lado lejano" y la variante de rodillas apoyadas.
- **Cómo se detecta dónde empieza/termina cada repetición**
  (`caida_minima_grados` en `config.py`, ver `segmentacion.py`): al
  principio se usaba un umbral de ángulo absoluto ("repetición = el codo
  bajó de 110°"), pero con las primeras grabaciones del equipo (2
  personas) se vio que el ángulo mínimo real de una repetición depende
  mucho del brazo/cuerpo de cada quien — el mismo umbral dejaba fuera
  repeticiones válidas de una persona (sobre todo las de
  `no_bajar_suficiente`, que por definición son las más cercanas al punto
  de partida) mientras funcionaba bien para otra. Se cambió a **caída
  relativa**: cuenta como repetición cualquier bajada de al menos
  `caida_minima_grados` desde el último punto más alto, sin importar en
  qué ángulo absoluto pase eso. Son valores iniciales razonables, no
  calibrados a fondo todavía — si a alguien del equipo le detecta muy
  pocas repeticiones en un video que sí tiene varias, avisen en el grupo
  antes de cambiar el valor por su cuenta.
- **El codo "se adelanta" en el curl se mide en 2D (desplazamiento
  horizontal en la imagen), no en profundidad real.** MediaPipe sí da una
  z aproximada, pero es menos confiable que x/y con una sola cámara; en
  vista lateral, que el codo se adelante SÍ se ve como desplazamiento
  horizontal, así que no hace falta la z.

## Datos: videos propios + dataset de apoyo

El taxonomía de errores (cadera caída vs. elevada, talones levantados,
codo que se adelanta, etc.) es específica de este póster — no existe un
dataset público ya etiquetado así. **La fuente principal son videos
propios del equipo**, grabados dos veces a propósito: repeticiones
correctas Y repeticiones con cada error específico, exagerado a propósito
para que quede claro (ver [GUIA_EQUIPO.md](GUIA_EQUIPO.md)). Es más fácil
de lo que suena: en vez de esperar a cometer el error por accidente,
cada quien hace el error adrede y lo etiqueta con la carpeta correcta.

Como apoyo opcional (más variedad de cuerpos/cámaras para la clase
"correcto" y para una clase genérica "algo está mal"), existe un dataset
público con push-up, sentadilla y curl de bíceps etiquetados "good"/"bad":
[A Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad
Form Labels](https://data.mendeley.com/datasets/kgbb3yn47p) (Mendeley
Data, 26 personas, vistas frontal/lateral/diagonal). No sustituye a los
videos propios con errores específicos, solo los complementa — ver
`importar_mendeley.py`.

## Archivos

| Archivo | Descripción |
|---|---|
| `Evaluador_Ejercicios.py` | Punto de entrada único (`extraer` / `entrenar` / `evaluar` / `tiempo-real`). |
| `config.py` | Puntos de MediaPipe, ángulos y umbrales por ejercicio, etiquetas, hiperparámetros. |
| `pose_utils.py` | Wrapper de MediaPipe Pose (Tasks API) y geometría (ángulos, lado visible, desviaciones con signo). |
| `caracteristicas.py` | Landmarks de un frame → señales del ejercicio; señales de una repetición → vector agregado. |
| `segmentacion.py` | Máquina de estados que detecta dónde empieza y termina cada repetición. |
| `modelo.py` | Ensamble CNN-1D híbrida + MLP (PyTorch): repetición → clase. |
| `extraer_caracteristicas.py` | `data/propios/.../*.mp4` → `data/caracteristicas/<ejercicio>.csv` (avisa videos con conteo de repeticiones raro). |
| `entrenar.py` | Entrena el modelo final de cada ejercicio con todas las repeticiones (lo que usa `tiempo-real`). |
| `evaluar.py` | Validación cruzada agrupada por serie (o `--por-persona`): reporte por clase y matriz de confusión. |
| `tiempo_real.py` | Evaluación en vivo con la webcam. |
| `importar_mendeley.py` | Importa el dataset de apoyo de Mendeley (manual, ver el propio archivo). |
| `data/propios/<ejercicio>/<etiqueta>/<persona>/` | Videos del equipo (no van a Git, ver `.gitignore`). |
| `data/caracteristicas/*.csv` | Características ya extraídas, por repetición (sí van a Git, son pequeños). |
| `salidas/` | Modelos entrenados (`.pt`, pesan poco, sí van a Git), métricas y matrices de confusión. |

## Cómo ejecutar

```powershell
# mediapipe no soporta Python 3.14 todavía -- usar Python 3.11
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

.venv\Scripts\python Evaluador_Ejercicios.py extraer                       # videos -> características (CSV)
.venv\Scripts\python Evaluador_Ejercicios.py entrenar                      # entrena los 3 modelos
.venv\Scripts\python Evaluador_Ejercicios.py evaluar                       # validación cruzada por serie
.venv\Scripts\python Evaluador_Ejercicios.py evaluar --por-persona         # deja fuera a una persona a la vez
.venv\Scripts\python Evaluador_Ejercicios.py tiempo-real --ejercicio sentadillas   # webcam en vivo
```

El modelo de MediaPipe Pose (`pose_landmarker_full.task`, ~30 MB) se
descarga solo la primera vez que se usa (`pose_utils.descargar_modelo_mediapipe`).

Cada subcomando acepta `--ejercicio {lagartijas,sentadillas,curl_biceps}`
para correr sobre uno solo en vez de los 3 (`tiempo-real` lo requiere
siempre, porque hay que decirle a la cámara qué ejercicio esperar).

No hay notebook de Colab para esta práctica: el modelo entrena sobre un
puñado de números por repetición — toma segundos en CPU, así que no hace
falta GPU gratuita ni notebook aparte.

**Cómo se mide**: `entrenar` usa todas las repeticiones porque el día de
la entrega se prueba con las mismas personas del equipo. Las métricas que
se reportan salen de `evaluar`, que entrena modelos nuevos dejando fuera
series completas (videos), así nunca se prueba con un video que el modelo
ya vio. `--por-persona` mide además qué tan bien generaliza a alguien
nuevo.
