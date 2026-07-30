"""
omr.py — Lector óptico de marcas para hojas Calco
Formato: 80 preguntas, opciones A-E, 4 columnas × 20 filas, tamaño carta
Hoja: hoja_preuc.html (215.9 × 279.4mm)
"""

import cv2
import numpy as np
from PIL import Image
import io
import subprocess
import tempfile
import os


def _es_heic(ruta: str) -> bool:
    """Detecta archivos HEIC/HEIF por extensión o magic bytes."""
    ext = os.path.splitext(ruta)[1].lower()
    if ext in ('.heic', '.heif'):
        return True
    try:
        with open(ruta, 'rb') as f:
            header = f.read(12)
        # Magic bytes HEIC: 'ftyp' en offset 4
        return header[4:8] == b'ftyp'
    except Exception:
        return False


def _convertir_heic_a_jpg(ruta_heic: str) -> str:
    """
    Convierte un archivo HEIC a JPEG usando ImageMagick (convert).
    Retorna la ruta del archivo JPEG temporal.
    """
    ruta_jpg = ruta_heic + '_converted.jpg'
    try:
        result = subprocess.run(
            ['convert', ruta_heic, ruta_jpg],
            capture_output=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(ruta_jpg):
            return ruta_jpg
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    # Fallback: heif-convert (si está disponible)
    try:
        result = subprocess.run(
            ['heif-convert', ruta_heic, ruta_jpg],
            capture_output=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(ruta_jpg):
            return ruta_jpg
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def leer_imagen(ruta: str) -> np.ndarray:
    """
    Carga una imagen en cualquier formato (JPEG, PNG, HEIC, WebP, etc.)
    y la retorna como array BGR de OpenCV.
    """
    # Intentar con OpenCV primero (JPEG/PNG rápido)
    img = cv2.imread(ruta)
    if img is not None:
        return img

    # Si es HEIC, convertir con ImageMagick
    if _es_heic(ruta):
        ruta_jpg = _convertir_heic_a_jpg(ruta)
        if ruta_jpg:
            try:
                img = cv2.imread(ruta_jpg)
                return img
            finally:
                if os.path.exists(ruta_jpg):
                    os.unlink(ruta_jpg)
        raise ValueError(f"No se pudo convertir HEIC: {ruta}. Asegúrate de que ImageMagick esté instalado con soporte HEIC.")

    # Fallback genérico: Pillow
    pil = Image.open(ruta).convert('RGB')
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# ── Dimensiones de trabajo ────────────────────────────────────────────────────
ANCHO = 1240   # px (carta horizontal)
ALTO  = 1754   # px (carta vertical)

# ── Coordenadas X relativas de burbujas A-E por columna ──────────────────────
# Cálculo: col_left + 13mm para A, +7mm por cada opción siguiente
# Dividido por 215.9mm para obtener valor relativo
#
# Col 1 (left=10mm):  A=23, B=30, C=37, D=44, E=51mm
# Col 2 (left=58.975mm): A=72, B=79, C=86, D=93, E=100mm
# Col 3 (left=107.95mm): A=121, B=128, C=135, D=142, E=149mm
# Col 4 (left=156.925mm): A=170, B=177, C=184, D=191, E=198mm

OPC_X = [
    [0.107, 0.139, 0.171, 0.204, 0.236],  # Col 1: Q1-20
    [0.333, 0.366, 0.398, 0.431, 0.463],  # Col 2: Q21-40
    [0.560, 0.593, 0.625, 0.657, 0.690],  # Col 3: Q41-60
    [0.787, 0.819, 0.852, 0.884, 0.917],  # Col 4: Q61-80
]

# ── Coordenadas Y relativas ───────────────────────────────────────────────────
# Area top=26mm, header=5.5mm, row height=11mm
# Fila 1 centro: 26 + 5.5 + 5.5 = 37mm  → 37/279.4 = 0.132
# Fila 20 centro: 37 + 19×11 = 246mm    → 246/279.4 = 0.880
Y_INICIO = 0.132
Y_FIN    = 0.880
N_FILAS  = 20   # preguntas por columna

# ── Parámetros de detección ───────────────────────────────────────────────────
RADIO_BURBUJA  = 12    # px en imagen normalizada (burbuja real ~15px de radio)
UMBRAL_MARCADO = 0.25  # fracción de píxeles oscuros para considerar marcada

OPCIONES = ['A', 'B', 'C', 'D', 'E']


# ── Detección de esquinas ─────────────────────────────────────────────────────
def detectar_esquinas(img_gris: np.ndarray):
    """
    Detecta los 4 marcadores cuadrados negros de las esquinas.
    Retorna array [TL, TR, BL, BR] o None si no encuentra 4.
    """
    _, binaria = cv2.threshold(img_gris, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for c in contornos:
        area = cv2.contourArea(c)
        if 500 < area < 8000:
            x, y, cw, ch = cv2.boundingRect(c)
            aspecto = cw / ch if ch > 0 else 0
            if 0.6 < aspecto < 1.6:
                candidatos.append((x + cw // 2, y + ch // 2))

    if len(candidatos) < 4:
        return None

    # Ordenar TL, TR, BL, BR
    candidatos.sort(key=lambda p: p[0] + p[1])
    tl = candidatos[0]
    br = candidatos[-1]
    candidatos.sort(key=lambda p: p[0] - p[1])
    tr = candidatos[-1]
    bl_candidates = [p for p in candidatos if p != tl and p != tr and p != br]
    bl = bl_candidates[0] if bl_candidates else candidatos[1]

    return np.array([tl, tr, bl, br], dtype=np.float32)


# ── Corrección de perspectiva ─────────────────────────────────────────────────
def corregir_perspectiva(imagen: np.ndarray) -> np.ndarray:
    """
    Usa los 4 marcadores de esquina para "aplanar" la hoja
    y normalizarla a ANCHO × ALTO píxeles.
    Sin marcadores: redimensiona directamente.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen.copy()
    esquinas = detectar_esquinas(gris)

    if esquinas is None:
        return cv2.resize(imagen, (ANCHO, ALTO))

    dst = np.array([
        [0,     0    ],
        [ANCHO, 0    ],
        [0,     ALTO ],
        [ANCHO, ALTO ],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(esquinas, dst)
    return cv2.warpPerspective(imagen, M, (ANCHO, ALTO))


# ── Lectura de una burbuja ────────────────────────────────────────────────────
def leer_burbuja(img_bin: np.ndarray, cx: int, cy: int, radio: int = RADIO_BURBUJA) -> float:
    """
    Dibuja un círculo en la posición dada y cuenta qué fracción
    de sus píxeles son oscuros (marcados con lápiz).
    """
    mascara = np.zeros(img_bin.shape, dtype=np.uint8)
    cv2.circle(mascara, (cx, cy), radio, 255, -1)
    region = cv2.bitwise_and(img_bin, img_bin, mask=mascara)
    pixeles_totales = np.sum(mascara > 0)
    pixeles_oscuros = np.sum(region == 0)
    return pixeles_oscuros / pixeles_totales if pixeles_totales > 0 else 0.0


# ── Procesamiento principal ───────────────────────────────────────────────────
def procesar_hoja(ruta_imagen: str) -> dict:
    """
    Procesa una foto de hoja de respuestas Calco (4 columnas × 20 filas).
    Retorna dict {1: 'A', 2: 'C', ..., 80: None} con la respuesta detectada
    (None = omitida).
    """
    img = leer_imagen(ruta_imagen)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}")

    # 1. Corregir perspectiva y normalizar a 1240×1754
    img_norm = corregir_perspectiva(img)

    # 2. Binarizar con umbral de Otsu (adaptativo al contraste de la foto)
    gris = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY) if len(img_norm.shape) == 3 else img_norm
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    respuestas = {}

    # 3. Recorrer 4 columnas × 20 filas
    for col_idx, xs in enumerate(OPC_X):
        offset_q = col_idx * N_FILAS   # 0, 20, 40, 60

        for fila in range(N_FILAS):
            num_pregunta = fila + 1 + offset_q

            # Posición Y interpolada entre primera y última fila
            y_rel = Y_INICIO + (Y_FIN - Y_INICIO) * fila / (N_FILAS - 1)
            cy = int(y_rel * ALTO)

            # Leer las 5 burbujas (A-E)
            marcados = []
            for i, x_rel in enumerate(xs):
                cx = int(x_rel * ANCHO)
                fraccion = leer_burbuja(binaria, cx, cy)
                if fraccion >= UMBRAL_MARCADO:
                    marcados.append(OPCIONES[i])

            if len(marcados) == 1:
                respuestas[num_pregunta] = marcados[0]
            elif len(marcados) == 0:
                respuestas[num_pregunta] = None          # omitida
            else:
                respuestas[num_pregunta] = marcados[0]   # doble marca → toma primera

    return respuestas


# ── Calificación ──────────────────────────────────────────────────────────────
def calcular_nota(respuestas: dict, pauta: dict, total_preguntas: int = 80) -> dict:
    """
    Compara respuestas con la pauta y calcula la nota en escala 1.0-7.0.
    Sin descuento por error (igual que PAES).
    Umbral de aprobación: 60% correctas → nota 4.0
    """
    correctas = incorrectas = omitidas = 0
    detalle = {}

    for q in range(1, total_preguntas + 1):
        resp     = respuestas.get(q)
        correcta = pauta.get(str(q)) or pauta.get(q)
        es_ok    = (resp is not None and resp == correcta)

        detalle[q] = {
            'respuesta': resp,
            'correcta':  correcta,
            'resultado': 'correcta' if es_ok else ('omitida' if resp is None else 'incorrecta'),
        }
        if es_ok:        correctas   += 1
        elif resp is None: omitidas  += 1
        else:              incorrectas += 1

    # Escala lineal 1-7 con quiebre en 60%
    fraccion = correctas / total_preguntas
    if   fraccion <= 0:    nota = 1.0
    elif fraccion >= 1:    nota = 7.0
    elif fraccion < 0.60:  nota = 1.0 + (fraccion / 0.60) * 3.0
    else:                  nota = 4.0 + ((fraccion - 0.60) / 0.40) * 3.0

    return {
        'correctas':   correctas,
        'incorrectas': incorrectas,
        'omitidas':    omitidas,
        'total':       total_preguntas,
        'nota':        round(nota, 1),
        'porcentaje':  round(fraccion * 100, 1),
        'detalle':     detalle,
    }
