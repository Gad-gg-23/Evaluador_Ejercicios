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
4. De cada repetición se saca un **vector de características agregadas**
   (mínimo, máximo, media y rango de cada señal durante la repetición — no
   los frames crudos). Un **MLP pequeño por ejercicio** (PyTorch) clasifica
   ese vector en `correcto` o uno de los errores del póster.

### Por qué así y no de otra forma

- **Características por repetición, no frames crudos ni una red
  convolucional sobre video**: el póster ya redujo el problema a ángulos de
  articulaciones, que es justo lo que hace manejable entrenar con pocos
  videos del equipo (a diferencia de una CNN/LSTM sobre píxeles, que
  necesitaría muchísimos más datos de los que un equipo de estudiantes
  puede grabar en unas semanas).
- **Un MLP por ejercicio, no un solo modelo para los 3**: cada ejercicio
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
- **Los umbrales de ángulo para detectar dónde empieza/termina cada
  repetición** (`umbral_superior`/`umbral_inferior` en `config.py`) se
  revisan con las grabaciones reales del equipo conforme van llegando.
  Lagartijas ya se recalibró con las primeras grabaciones (el valor de
  referencia original, 110°, era demasiado estricto y dejaba fuera
  repeticiones válidas, sobre todo las de `no_bajar_suficiente`; ver el
  commit que subió ese ajuste). Sentadillas y curl de bíceps siguen con el
  valor de referencia inicial (funcionó bien con la primera persona, pero
  puede necesitar ajuste con más gente) — si a alguien del equipo le
  detecta muy pocas repeticiones en un video que sí tiene varias, avisen en
  el grupo antes de cambiar el umbral por su cuenta.
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
| `modelo.py` | MLP (PyTorch) característica-de-repetición → clase. |
| `extraer_caracteristicas.py` | `data/propios/.../*.mp4` → `data/caracteristicas/<ejercicio>.csv`. |
| `entrenar.py` | Entrena el MLP de cada ejercicio (split agrupado por persona, early stopping). |
| `evaluar.py` | Métricas y matriz de confusión sobre la partición de prueba. |
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
.venv\Scripts\python Evaluador_Ejercicios.py entrenar                      # entrena los 3 MLP
.venv\Scripts\python Evaluador_Ejercicios.py evaluar                       # métricas + matriz de confusión
.venv\Scripts\python Evaluador_Ejercicios.py tiempo-real --ejercicio sentadillas   # webcam en vivo
```

El modelo de MediaPipe Pose (`pose_landmarker_full.task`, ~30 MB) se
descarga solo la primera vez que se usa (`pose_utils.descargar_modelo_mediapipe`).

Cada subcomando acepta `--ejercicio {lagartijas,sentadillas,curl_biceps}`
para correr sobre uno solo en vez de los 3 (`tiempo-real` lo requiere
siempre, porque hay que decirle a la cámara qué ejercicio esperar).

No hay notebook de Colab para esta práctica: el MLP entrena sobre un
puñado de números por repetición — toma segundos en CPU, así que no hace
falta GPU gratuita ni notebook aparte.
