# Guía paso a paso — Evaluador de ejercicios (ESCOM)

Esta guía es para **cualquier integrante del equipo** que vaya a grabar
videos, entrenar, o probar el evaluador en vivo. No hace falta cuenta en
ningún servicio externo — solo Python y una cámara.

## Resumen ultra-corto

```
Cada quien:  graba repeticiones correctas + repeticiones con cada error (adrede)
                 -> las guarda en la carpeta correcta -> extrae características -> entrena
                 -> git add data/caracteristicas + salidas -> git push
```

---

## Parte 1 — Prepara tu máquina (una sola vez)

MediaPipe **no soporta Python 3.14 todavía**, así que esta práctica
necesita Python 3.11 (revisa con `py -0p` qué versiones tienes instaladas).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Si `py -3.11` no encuentra nada, instala Python 3.11 desde
[python.org/downloads](https://www.python.org/downloads/) (marca "Add to
PATH" al instalar).

---

## Parte 2 — Graba tus videos

### 2.1 Cámara y encuadre

- **Vista lateral**, cuerpo completo (de pies a cabeza) dentro del cuadro,
  igual que el póster. Celular en horizontal, apoyado o con tripié — que no
  se mueva durante la serie.
- Luz de frente (no a contraluz) y fondo que no se confunda con el cuerpo.
- **Lagartijas**: a MediaPipe le cuesta más el cuerpo horizontal. Si tienes
  tripié, un poco en diagonal (ni al ras del piso ni muy de arriba) ayuda
  más que el perfil perfecto.
- **Sentadillas** y **curl de bíceps**: de pie, perfil normal a la cámara.

### 2.2 Qué grabar: correcto Y cada error, a propósito

No hace falta esperar a equivocarte por accidente. Para cada ejercicio,
graba varias series cortas (5-8 repeticiones) así:

| Ejercicio | Carpeta `etiqueta` | Cómo hacerlo |
|---|---|---|
| **Lagartijas** | `correcto` | Técnica normal, cadera alineada. |
| | `cadera_caida` | Deja caer la cadera a propósito (como una "V" hacia abajo). |
| | `cadera_elevada` | Sube la cadera de más (como un "pico"). |
| | `no_bajar_suficiente` | Repeticiones cortas, casi sin flexionar el codo. |
| **Sentadillas** | `correcto` | Técnica normal, baja a paralelo. |
| | `no_bajar_suficiente` | Sentadillas de cuarto de rango, muy poco profundas. |
| | `inclinarse_demasiado` | Inclina mucho el torso hacia adelante al bajar. |
| | `talones_levantados` | Despega los talones del piso al bajar (ponte de puntas al final). |
| **Curl de bíceps** | `correcto` | Técnica normal, codo pegado al torso. |
| | `balancear_torso` | Usa impulso del cuerpo (échate para atrás/adelante) en vez de solo el brazo. |
| | `codo_se_adelanta` | Deja que el codo se adelante del torso al subir el peso. |
| | `no_completar_movimiento` | Sube el peso solo un poco, sin flexionar bien el codo. |

Entre más personas graben las 4 (o 3) etiquetas de un ejercicio, más
fácil es para el modelo aprender a distinguirlas — no dejes que una sola
persona grabe "todo lo malo" y otra "todo lo bueno".

### 2.2.1 ¿Cuántos videos grabo?

**Antes de que todo el equipo grabe en serio, que una sola persona grabe
un piloto:** 1 serie de cada etiqueta de un solo ejercicio (ej.
sentadillas) y corra `extraer` (Parte 3). Los umbrales que detectan dónde
empieza/termina una repetición (`config.py`) todavía no están calibrados
con cámara real — mejor descubrir ahí si hay que ajustarlos que que las
otras 4 personas repitan el mismo problema.

Ya calibrado, la meta por persona es:

| | Cantidad |
|---|---|
| Ejercicios × etiquetas | 3 ejercicios × 4 etiquetas = 12 combinaciones |
| Series por combinación | **mínimo 2** series de 6-8 repeticiones cada una |
| Videos por persona | 24 (12 combinaciones × 2 series) |
| Videos del equipo (5 personas) | ≈120 en total |

¿Por qué mínimo 2 y no 1? Con una sola serie por etiqueta, si esa
grabación sale rara (mala luz, ángulo raro, una repetición atípica), esa
etiqueta completa de esa persona queda con un solo dato de apoyo. Con 2
series ya hay algo de variación aunque sea de la misma persona el mismo
día. No tiene que ser exacto — si alguien graba 1 o 3 de alguna etiqueta
no pasa nada — pero que nadie se quede en 1 sola serie por etiqueta, y que
**las 5 personas aporten las 4 etiquetas**, no que una sola grabe "todo lo
malo" y el resto solo "correcto" (si no, el modelo aprende a distinguir
personas, no técnica).

Equipo necesario: **ninguno** para lagartijas/sentadillas. Para curl de
bíceps, algo de peso (garrafón, mochila con libros, botellas con agua) —
que todo el equipo use algo parecido, para no meter demasiada variación
extra.

### 2.3 Guarda los videos en la carpeta correcta

```
data/propios/<ejercicio>/<etiqueta>/<tu_nombre>/
```

Ejemplo (Juan grabando lagartijas):

```
data/propios/lagartijas/correcto/juan/serie_01.mp4
data/propios/lagartijas/correcto/juan/serie_02.mp4
data/propios/lagartijas/cadera_caida/juan/serie_01.mp4
data/propios/lagartijas/no_bajar_suficiente/juan/serie_01.mp4
```

`<ejercicio>` es exactamente `lagartijas`, `sentadillas` o `curl_biceps`;
`<etiqueta>` exactamente como en la tabla de arriba (son los nombres de
carpeta que ya existen en `config.py` — si se escribe distinto, ese video
simplemente no se usa). `<tu_nombre>` puede ser tu nombre o apodo, lo que
sea consistente contigo mismo entre series.

Esta carpeta **no se sube a GitHub** (son videos pesados y es tu imagen);
lo que sí se sube son las características ya extraídas (números, no
video) y el modelo entrenado.

---

## Parte 3 — Extrae características y entrena

**Por qué este orden importa:** tus videos nunca se suben a GitHub (son
pesados y es tu imagen) — lo único que viaja por GitHub es
`data/caracteristicas/<ejercicio>.csv`, que son solo números (una fila por
repetición). Como tú nunca tienes en tu máquina los videos de tus
compañeros, `extraer` NO reescribe el CSV completo: conserva las filas de
quien no tengas localmente y solo recalcula las tuyas. Pero para que eso
funcione, necesitas partir del CSV más reciente — por eso el `git pull` va
**primero**, antes de extraer.

```powershell
git pull                                                   # trae las características más recientes del equipo
.venv\Scripts\python Evaluador_Ejercicios.py extraer       # tus videos -> se fusionan con el CSV del equipo
.venv\Scripts\python Evaluador_Ejercicios.py entrenar      # entrena los 3 MLP con TODO el CSV (tú + equipo)
.venv\Scripts\python Evaluador_Ejercicios.py evaluar       # revisa métricas y matriz de confusión en salidas/
```

`extraer` imprime cuántas repeticiones detectó por video — si un video de
8 repeticiones solo detecta 2 o 3, probablemente los umbrales de
`config.py` (`umbral_superior`/`umbral_inferior` de ese ejercicio) no
cuadran con tu ángulo de cámara o tu rango de movimiento; avisa en el
grupo para ajustarlos juntos, no cada quien por su lado.

`entrenar` avisa si todavía falta alguna etiqueta o si hay muy pocas
personas — es normal al principio, mientras el equipo va subiendo videos.
El modelo que entrena cada quien ya incluye los datos de todo el equipo
que esté en el CSV al momento de entrenar, aunque los videos en sí nunca
salgan de la compu de cada quien.

Para compartir tu aporte:

```powershell
git add data/caracteristicas salidas
git commit -m "Agrega características y reentrena con los videos de <tu nombre>"
git push
```

Si entre tu `git pull` y tu `git push` alguien más ya subió su parte, Git
va a pedir que hagas `git pull` otra vez antes de dejarte subir (conflicto
normal) — vuelve a correr `extraer` y `entrenar` después de ese segundo
pull para no perder lo que acaba de subir esa persona.

---

## Parte 4 — Pruébalo en vivo

```powershell
.venv\Scripts\python Evaluador_Ejercicios.py tiempo-real --ejercicio sentadillas
```

Colócate como en tus videos de grabación (vista lateral, cuerpo completo
en cuadro). La ventana muestra el esqueleto detectado, el contador de
repeticiones y, al terminar cada una, si salió `correcto` o qué error
detectó. `q` para salir.

---

## Problemas comunes

- **`ModuleNotFoundError: No module named 'mediapipe'`**: estás usando el
  Python equivocado. Activa el venv (`.venv\Scripts\python`, no `python`
  a secas) o revisa que se haya creado con Python 3.11.
- **La cámara no abre / ventana en negro**: otro programa (Zoom, Teams)
  puede tener la cámara ocupada; ciérralo. Si tienes más de una cámara,
  prueba `--camara 1`.
- **`extraer` detecta 0 repeticiones en un video que sí tiene**: revisa
  encuadre (cuerpo completo, vista lateral) y luz; si el encuadre se ve
  bien, es cosa de ajustar los umbrales de ese ejercicio en `config.py`
  (avisar al equipo antes de cambiarlos, afecta a todos).
- **El modelo dice "correcto" a todo o se equivoca mucho**: normal con
  pocos videos. Hace falta más variedad (más personas, más series por
  etiqueta) antes de confiar en las métricas — `entrenar.py` ya avisa
  cuando hay muy pocas personas para una división train/val/prueba
  confiable.
