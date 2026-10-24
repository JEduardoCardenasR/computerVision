# Asistencia con reconocimiento facial

Sistema sencillo que detecta un rostro, lo compara con las personas registradas y anota la asistencia. Si alguien registrado no aparece, la consola simula un aviso a padres o supervisores (solo un mensaje, no envía nada).

## Qué necesitas

- Python 3.10, 3.11 o 3.12
- Una cámara web, **o** una foto clara de frente (JPG o PNG)

## Cómo ejecutarlo

Abre una terminal **en esta carpeta** (la que contiene `asistencia.py`).

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python asistencia.py
```

### Mac o Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python asistencia.py
```

La primera vez `pip` puede tardar un poco: instala OpenCV.

## Cómo probarlo

1. Elige **1** y escribe un nombre. Mira de frente a la cámara hasta que junte 20 fotos. **ESC** cancela.
2. Elige **3**. Vuelve a mirar la cámara. Si te reconoce, imprime `Asistencia registrada: tu nombre` y dibuja un recuadro verde.
3. Pulsa **Q**. Abajo verás quién llegó y un aviso simulado para quien no llegó.
4. Elige **5** para ver la lista del día. El archivo queda en `datos/asistencia.csv`.

### Sin cámara

1. Elige **2**, escribe el nombre y pega la ruta de una foto. Ejemplo: `C:\Users\Ana\Desktop\ana.jpg`
2. Elige **4** y usa otra foto de la misma persona (o la misma).
3. Elige **5** para ver el registro.

Con una sola foto el reconocimiento es más débil que con la cámara. Si sale `Desconocido`, registra de nuevo con más luz o sube `UMBRAL` en `asistencia.py` (está al inicio del archivo; un número más alto acepta parecidos más flojos).

## Controles

| Tecla | Dónde | Qué hace |
| --- | --- | --- |
| ESC | Registro con cámara | Cancela |
| Q | Asistencia con cámara | Termina y muestra faltas |
| Cualquier tecla | Ventana de una foto | Cierra la foto |

El número junto al nombre es la distancia. Más bajo significa que se parece más. Si pasa de `UMBRAL` (70), se marca como desconocido y no se anota asistencia.

## Si algo falla

- **No abre la cámara:** cierra otras apps que la usen, o prueba las opciones 2 y 4 con una foto.
- **Siempre dice Desconocido:** registra otra vez con más luz, de frente. Si sigue igual, en `asistencia.py` sube `UMBRAL` de 70 a 90.
- **Error al abrir archivos y la carpeta tiene acentos:** el programa ya usa el nombre corto de Windows. Si aun así falla, copia el proyecto a `C:\asistencia` y ejecútalo desde ahí.
- La carpeta `computerVision` es otro ejemplo. Este proyecto es `asistencia.py`.

## Cómo está hecho

- **Detección:** Haar Cascade de OpenCV. Encuentra el rectángulo del rostro.
- **Reconocimiento:** LBPH (`opencv-contrib-python`). Compara el rostro con las fotos de `datos/rostros/`.
- **Asistencia:** una fila por persona al día en `datos/asistencia.csv`.
- **Aviso:** si alguien está registrado y no tiene fila hoy, se imprime `Notificando a padres/supervisor: ...`. No se envía correo ni mensaje real.
