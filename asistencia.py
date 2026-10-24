# -*- coding: utf-8 -*-
"""
Asistencia automatica con reconocimiento facial.

1. La camara (o una foto) busca un rostro con un Haar Cascade de OpenCV.
2. Esas fotos se comparan con LBPH: un numero chico significa "es la misma persona".
3. Si coincide, se anota la asistencia en datos/asistencia.csv.
4. Quien este registrado y no aparezca recibe un aviso simulado en la consola.
"""

import csv
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

CARPETA = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(CARPETA, "datos")
ROSTROS = os.path.join(DATOS, "rostros")
MODELO = os.path.join(DATOS, "modelo.yml")
ETIQUETAS = os.path.join(DATOS, "etiquetas.json")
ASISTENCIA = os.path.join(DATOS, "asistencia.csv")

# Distancia LBPH. Mas bajo = mas parecido. Sube este numero si te marca "Desconocido".
UMBRAL = 70
TAMANO = (200, 200)
FOTOS_POR_PERSONA = 20


def ruta_opencv(ruta):
    """OpenCV en Windows no abre carpetas con acentos. Usa el nombre corto."""
    ruta = os.path.abspath(ruta)
    if os.name != "nt" or ruta.isascii():
        return ruta
    import ctypes

    buf = ctypes.create_unicode_buffer(32768)
    objetivo = ruta if os.path.exists(ruta) else os.path.dirname(ruta)
    n = ctypes.windll.kernel32.GetShortPathNameW(objetivo, buf, len(buf))
    if not n or not buf.value.isascii():
        print("Esta carpeta tiene acentos y OpenCV no puede usarla.")
        print(r"Mueve el proyecto a una ruta simple, por ejemplo C:\asistencia")
        sys.exit(1)
    if os.path.exists(ruta):
        return buf.value
    return os.path.join(buf.value, os.path.basename(ruta))


def plano(mensaje):
    """La ventana de OpenCV solo dibuja letras sin acento."""
    return str(mensaje).translate(str.maketrans("áéíóúÁÉÍÓÚñÑüÜ", "aeiouAEIOUnNuU"))


def preparar():
    os.makedirs(ROSTROS, exist_ok=True)
    if not hasattr(cv2, "face"):
        print("Falta el reconocedor de rostros.")
        print("En el entorno virtual ejecuta:")
        print("  pip uninstall -y opencv-python opencv-python-headless")
        print("  pip install -r requirements.txt")
        sys.exit(1)


def detector():
    ruta = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(ruta):
        print("No encontre el detector de rostros.")
        print("Vuelve a instalar con: pip install -r requirements.txt")
        sys.exit(1)
    clasificador = cv2.CascadeClassifier(ruta_opencv(ruta))
    if clasificador.empty():
        print("No pude cargar el detector de rostros.")
        sys.exit(1)
    return clasificador


DETECTAR = None


def buscar_rostros(gris):
    global DETECTAR
    if DETECTAR is None:
        DETECTAR = detector()
    rectangulos = DETECTAR.detectMultiScale(
        gris, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )
    if rectangulos is None or len(rectangulos) == 0:
        return []
    return list(rectangulos)


def recortar(gris, rect):
    x, y, w, h = [int(v) for v in rect]
    cara = gris[y : y + h, x : x + w]
    if cara.size == 0:
        return None
    return cv2.resize(cara, TAMANO)


def leer_imagen(ruta, gris=False):
    """Lee una imagen aunque la ruta tenga acentos (OpenCV en Windows falla si no)."""
    try:
        datos = np.fromfile(ruta, dtype=np.uint8)
    except OSError:
        return None
    if datos.size == 0:
        return None
    bandera = cv2.IMREAD_GRAYSCALE if gris else cv2.IMREAD_COLOR
    return cv2.imdecode(datos, bandera)


def guardar_imagen(ruta, imagen):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", imagen)
    if not ok:
        return False
    buffer.tofile(ruta)
    return True


def nombre_valido(texto):
    nombre = " ".join(texto.strip().split())
    for caracter in '<>:"/\\|?*':
        nombre = nombre.replace(caracter, "")
    return nombre


def ruta_valida(texto):
    return texto.strip().strip('"').strip("'")


def guardar_rostro(nombre, cara):
    carpeta = os.path.join(ROSTROS, nombre)
    os.makedirs(carpeta, exist_ok=True)
    cuantas = len([f for f in os.listdir(carpeta) if f.lower().endswith(".jpg")])
    destino = os.path.join(carpeta, f"{cuantas + 1}.jpg")
    guardar_imagen(destino, cara)


def entrenar():
    imagenes = []
    etiquetas = []
    nombres = {}
    personas = [
        d for d in sorted(os.listdir(ROSTROS)) if os.path.isdir(os.path.join(ROSTROS, d))
    ]
    for indice, nombre in enumerate(personas):
        nombres[indice] = nombre
        carpeta = os.path.join(ROSTROS, nombre)
        for archivo in sorted(os.listdir(carpeta)):
            if not archivo.lower().endswith(".jpg"):
                continue
            cara = leer_imagen(os.path.join(carpeta, archivo), gris=True)
            if cara is None:
                continue
            if cara.shape != TAMANO:
                cara = cv2.resize(cara, TAMANO)
            imagenes.append(cara)
            etiquetas.append(indice)

    if not imagenes:
        print("No hay rostros guardados para entrenar.")
        return False

    reconocedor = cv2.face.LBPHFaceRecognizer_create()
    reconocedor.train(imagenes, np.array(etiquetas))
    os.makedirs(DATOS, exist_ok=True)
    reconocedor.write(ruta_opencv(MODELO))
    with open(ETIQUETAS, "w", encoding="utf-8") as archivo:
        json.dump(nombres, archivo, ensure_ascii=False, indent=2)
    print(f"Listo. Entrene {len(imagenes)} fotos de {len(nombres)} persona(s).")
    return True


def cargar():
    if not os.path.exists(MODELO) or not os.path.exists(ETIQUETAS):
        print("Primero registra al menos una persona (opcion 1 o 2).")
        return None, None
    reconocedor = cv2.face.LBPHFaceRecognizer_create()
    reconocedor.read(ruta_opencv(MODELO))
    with open(ETIQUETAS, encoding="utf-8") as archivo:
        crudo = json.load(archivo)
    nombres = {int(k): v for k, v in crudo.items()}
    return reconocedor, nombres


def abrir_camara():
    if os.name == "nt":
        camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if camara.isOpened():
            return camara
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("No pude abrir la camara. Usa la opcion de foto.")
        return None
    return camara


def texto(frame, mensaje, y, color=(255, 255, 255)):
    cv2.putText(frame, plano(mensaje), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def registrar_camara():
    nombre = nombre_valido(input("Nombre de la persona: "))
    if not nombre:
        print("El nombre esta vacio.")
        return
    camara = abrir_camara()
    if camara is None:
        return

    print("Mira de frente a la camara. ESC cancela.")
    guardadas = 0
    visto = 0
    try:
        while guardadas < FOTOS_POR_PERSONA:
            ok, frame = camara.read()
            if not ok:
                print("La camara no entrego imagen.")
                break
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rostros = buscar_rostros(gris)
            if rostros:
                rect = max(rostros, key=lambda r: int(r[2]) * int(r[3]))
                x, y, w, h = [int(v) for v in rect]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                visto += 1
                if visto % 5 == 0:
                    cara = recortar(gris, rect)
                    if cara is not None:
                        guardar_rostro(nombre, cara)
                        guardadas += 1
                texto(frame, f"{nombre}: {guardadas}/{FOTOS_POR_PERSONA}", 30, (0, 255, 0))
            else:
                texto(frame, "No veo un rostro", 30, (0, 0, 255))
            texto(frame, "ESC cancela", 60)
            cv2.imshow("Registro", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camara.release()
        cv2.destroyAllWindows()

    if guardadas == 0:
        print("No se guardo ninguna foto.")
        return
    print(f"Se guardaron {guardadas} fotos de {nombre}.")
    entrenar()


def registrar_foto():
    nombre = nombre_valido(input("Nombre de la persona: "))
    if not nombre:
        print("El nombre esta vacio.")
        return
    ruta = ruta_valida(input("Ruta de la foto: "))
    imagen = leer_imagen(ruta)
    if imagen is None:
        print("No pude abrir esa foto.")
        return
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    rostros = buscar_rostros(gris)
    if not rostros:
        print("No hay un rostro claro en esa foto. Usa una cara de frente y con luz.")
        return
    rect = max(rostros, key=lambda r: int(r[2]) * int(r[3]))
    cara = recortar(gris, rect)
    if cara is None:
        print("No pude recortar el rostro.")
        return
    guardar_rostro(nombre, cara)
    x, y, w, h = [int(v) for v in rect]
    cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
    texto(imagen, plano(nombre), max(20, y - 10), (0, 255, 0))
    print(f"Rostro de {nombre} guardado.")
    entrenar()
    mostrar("Registro desde foto (cualquier tecla cierra)", imagen)


def identificar(reconocedor, nombres, cara):
    indice, distancia = reconocedor.predict(cara)
    nombre = nombres.get(int(indice), "Desconocido")
    if distancia > UMBRAL:
        return "Desconocido", distancia
    return nombre, distancia


def marcar(nombre, ya):
    if nombre == "Desconocido" or nombre in ya:
        return False
    nuevo = not os.path.exists(ASISTENCIA)
    ahora = datetime.now()
    with open(ASISTENCIA, "a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        if nuevo:
            escritor.writerow(["fecha", "hora", "nombre"])
        escritor.writerow([ahora.strftime("%Y-%m-%d"), ahora.strftime("%H:%M:%S"), nombre])
    ya.add(nombre)
    print(f"Asistencia registrada: {nombre} ({ahora.strftime('%H:%M:%S')})")
    return True


def asistieron_hoy():
    hoy = datetime.now().strftime("%Y-%m-%d")
    presentes = set()
    if not os.path.exists(ASISTENCIA):
        return presentes
    with open(ASISTENCIA, newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            if fila.get("fecha") == hoy:
                presentes.add(fila.get("nombre", ""))
    return presentes


def personas_registradas():
    if not os.path.isdir(ROSTROS):
        return []
    return sorted(
        d for d in os.listdir(ROSTROS) if os.path.isdir(os.path.join(ROSTROS, d))
    )


def avisar_ausentes():
    """Simula el aviso a padres o supervisores. No envia nada de verdad."""
    registrados = personas_registradas()
    if not registrados:
        print("No hay personas registradas.")
        return
    presentes = asistieron_hoy()
    print("")
    print("Presentes hoy:", ", ".join(sorted(presentes)) if presentes else "(nadie)")
    print("Simulacion de avisos:")
    faltaron = [n for n in registrados if n not in presentes]
    if not faltaron:
        print("  Todos los registrados asistieron.")
        return
    for nombre in faltaron:
        print(f"  Notificando a padres/supervisor: {nombre} no asistio.")


def tomar_asistencia():
    reconocedor, nombres = cargar()
    if reconocedor is None:
        return
    camara = abrir_camara()
    if camara is None:
        return

    print("Apunta la camara a las personas. Q termina y muestra quien falto.")
    ya = asistieron_hoy()
    try:
        while True:
            ok, frame = camara.read()
            if not ok:
                print("La camara no entrego imagen.")
                break
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for rect in buscar_rostros(gris):
                cara = recortar(gris, rect)
                if cara is None:
                    continue
                nombre, distancia = identificar(reconocedor, nombres, cara)
                x, y, w, h = [int(v) for v in rect]
                color = (0, 0, 255) if nombre == "Desconocido" else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    plano(f"{nombre} ({distancia:.0f})"),
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
                marcar(nombre, ya)
            texto(frame, "Q termina", 30)
            cv2.imshow("Asistencia", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camara.release()
        cv2.destroyAllWindows()
    avisar_ausentes()


def reconocer_foto():
    reconocedor, nombres = cargar()
    if reconocedor is None:
        return
    ruta = ruta_valida(input("Ruta de la foto: "))
    imagen = leer_imagen(ruta)
    if imagen is None:
        print("No pude abrir esa foto.")
        return
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    rostros = buscar_rostros(gris)
    if not rostros:
        print("No hay un rostro claro en esa foto.")
        return
    ya = asistieron_hoy()
    for rect in rostros:
        cara = recortar(gris, rect)
        if cara is None:
            continue
        nombre, distancia = identificar(reconocedor, nombres, cara)
        x, y, w, h = [int(v) for v in rect]
        color = (0, 0, 255) if nombre == "Desconocido" else (0, 255, 0)
        cv2.rectangle(imagen, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            imagen,
            plano(f"{nombre} ({distancia:.0f})"),
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        print(f"{nombre}  (distancia {distancia:.0f}, umbral {UMBRAL})")
        marcar(nombre, ya)
    mostrar("Resultado (cualquier tecla cierra)", imagen)


def ver_asistencia():
    if not os.path.exists(ASISTENCIA):
        print("Todavia no hay registros.")
        avisar_ausentes()
        return
    hoy = datetime.now().strftime("%Y-%m-%d")
    print(f"Asistencia del {hoy}:")
    hubo = False
    with open(ASISTENCIA, newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            if fila.get("fecha") == hoy:
                hubo = True
                print(f"  {fila['hora']}  {fila['nombre']}")
    if not hubo:
        print("  (sin registros hoy)")
    avisar_ausentes()


def mostrar(titulo, frame):
    try:
        cv2.imshow(titulo, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("No se abrio la ventana. El resultado quedo escrito arriba.")


def menu():
    preparar()
    opciones = {
        "1": registrar_camara,
        "2": registrar_foto,
        "3": tomar_asistencia,
        "4": reconocer_foto,
        "5": ver_asistencia,
    }
    while True:
        print("")
        print("========================================")
        print("  ASISTENCIA POR RECONOCIMIENTO FACIAL")
        print("========================================")
        print("1. Registrar persona con camara")
        print("2. Registrar persona con una foto")
        print("3. Tomar asistencia con camara")
        print("4. Reconocer una foto")
        print("5. Ver asistencia de hoy")
        print("0. Salir")
        eleccion = input("Opcion: ").strip()
        if eleccion == "0":
            print("Listo.")
            break
        accion = opciones.get(eleccion)
        if accion is None:
            print("Opcion no valida.")
            continue
        accion()


if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\nCancelado.")
