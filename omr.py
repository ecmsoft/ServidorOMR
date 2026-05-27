"""
omr.py — Lector óptico de marcas para hoja de respuestas Preuniversitario UC
Formato: 80 preguntas, opciones A-E, 4 marcadores de esquina
"""

import cv2
import numpy as np
from typing import Optional

# ── Dimensiones de trabajo (A4 @ 150 DPI aprox) ──────────────────────────────
ANCHO = 1240
ALTO  = 1754

# ── Coordenadas relativas de burbujas (0.0 – 1.0) ────────────────────────────
# Columna izquierda: preguntas 1-40
OPC_X_IZQ = [0.090, 0.128, 0.166, 0.204, 0.242]   # A B C D E

# Columna derecha: preguntas 41-80
OPC_X_DER = [0.548, 0.586, 0.624, 0.662, 0.700]   # A B C D E

# Rango vertical de preguntas (primera fila → última fila)
Y_INICIO = 0.182   # fila pregunta 1 / 41
Y_FIN    = 0.930   # fila pregunta 40 / 80

RADIO_BURBUJA  = 13    # píxeles en imagen normalizada
UMBRAL_MARCADO = 0.25  # fracción de píxeles oscuros para considerar marcado

OPCIONES = ['A', 'B', 'C', 'D', 'E']


# ── Detección de esquinas (marcadores negros en las 4 esquinas) ───────────────
def detectar_esquinas(img_gris: np.ndarray):
    """Detecta los 4 marcadores de esquina y retorna sus centroides."""
    _, binaria = cv2.threshold(img_gris, 60, 255, cv2.THRESH_BINARY_INV)

    # Morfología para limpiar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img_gris.shape
    candidatos = []
    for c in contornos:
        area = cv2.contourArea(c)
        if 500 < area < 8000:
            x, y, cw, ch = cv2.boundingRect(c)
            aspecto = cw / ch if ch > 0 else 0
            if 0.6 < aspecto < 1.6:
                cx = x + cw // 2
                cy = y + ch // 2
                candidatos.append((cx, cy))

    if len(candidatos) < 4:
        return None

    # Ordenar: TL, TR, BL, BR
    candidatos.sort(key=lambda p: p[0] + p[1])
    tl = candidatos[0]
    candidatos.sort(key=lambda p: -p[0] + p[1])
    tr_bl = candidatos
    candidatos.sort(key=lambda p: p[0] - p[1])
    tr = candidatos[0]
    candidatos.sort(key=lambda p: p[0] + p[1])
    br = candidatos[-1]
    bl_candidates = [p for p in candidatos if p != tl and p != tr and p != br]
    bl = bl_candidates[0] if bl_candidates else candidatos[1]

    return np.array([tl, tr, bl, br], dtype=np.float32)


def corregir_perspectiva(imagen: np.ndarray) -> np.ndarray:
    """Aplica corrección de perspectiva usando los marcadores de esquina."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen.copy()

    esquinas = detectar_esquinas(gris)
    if esquinas is None:
        # Sin marcadores: redimensionar directamente
        return cv2.resize(imagen, (ANCHO, ALTO))

    dst = np.array([
        [0,     0    ],
        [ANCHO, 0    ],
        [0,     ALTO ],
        [ANCHO, ALTO ],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(esquinas, dst)
    corregida = cv2.warpPerspective(imagen, M, (ANCHO, ALTO))
    return corregida


# ── Lectura de una burbuja ────────────────────────────────────────────────────
def leer_burbuja(img_bin: np.ndarray, cx: int, cy: int, radio: int = RADIO_BURBUJA) -> float:
    """Retorna fracción de píxeles oscuros en el círculo de la burbuja."""
    mascara = np.zeros(img_bin.shape, dtype=np.uint8)
    cv2.circle(mascara, (cx, cy), radio, 255, -1)
    region = cv2.bitwise_and(img_bin, img_bin, mask=mascara)
    pixeles_totales = np.sum(mascara > 0)
    pixeles_oscuros = np.sum(region == 0)
    return pixeles_oscuros / pixeles_totales if pixeles_totales > 0 else 0.0


# ── Procesamiento principal ───────────────────────────────────────────────────
def procesar_hoja(ruta_imagen: str) -> dict:
    """
    Procesa una foto de hoja de respuestas.
    Retorna dict con respuestas detectadas por pregunta (1-80).
    """
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}")

    # 1. Corrección de perspectiva
    img_norm = corregir_perspectiva(img)

    # 2. Binarización
    gris = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY) if len(img_norm.shape) == 3 else img_norm
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    respuestas = {}   # {1: 'A', 2: None, ...}

    # 3. Leer burbujas por columna
    for col, (xs, rango_y, offset_q) in enumerate([
        (OPC_X_IZQ, (Y_INICIO, Y_FIN), 0),   # preguntas 1-40
        (OPC_X_DER, (Y_INICIO, Y_FIN), 40),  # preguntas 41-80
    ]):
        n_preguntas = 40
        for fila in range(n_preguntas):
            num_pregunta = fila + 1 + offset_q
            y_rel = rango_y[0] + (rango_y[1] - rango_y[0]) * fila / (n_preguntas - 1)
            cy = int(y_rel * ALTO)

            marcados = []
            for i, x_rel in enumerate(xs):
                cx = int(x_rel * ANCHO)
                fraccion = leer_burbuja(binaria, cx, cy)
                if fraccion >= UMBRAL_MARCADO:
                    marcados.append(OPCIONES[i])

            if len(marcados) == 1:
                respuestas[num_pregunta] = marcados[0]
            elif len(marcados) == 0:
                respuestas[num_pregunta] = None   # sin marcar
            else:
                respuestas[num_pregunta] = marcados[0]  # toma la primera si hay doble marca

    return respuestas


# ── Calificación ─────────────────────────────────────────────────────────────
def calcular_nota(respuestas: dict, pauta: dict, total_preguntas: int = 80) -> dict:
    """
    Calcula correctas, incorrectas, omitidas y nota (escala 1.0 – 7.0).
    Fórmula PSU/PAES: sin descuento por error.
    Umbral de aprobación: 60% → nota 4.0
    """
    correctas  = 0
    incorrectas = 0
    omitidas   = 0

    detalle = {}
    for q in range(1, total_preguntas + 1):
        resp      = respuestas.get(q)
        correcta  = pauta.get(str(q)) or pauta.get(q)
        es_correcta = (resp is not None and resp == correcta)
        detalle[q] = {
            'respuesta': resp,
            'correcta':  correcta,
            'resultado': 'correcta' if es_correcta else ('omitida' if resp is None else 'incorrecta'),
        }
        if es_correcta:
            correctas += 1
        elif resp is None:
            omitidas += 1
        else:
            incorrectas += 1

    # Nota: escala lineal 1.0-7.0, umbral 60%
    fraccion = correctas / total_preguntas
    if fraccion <= 0:
        nota = 1.0
    elif fraccion >= 1:
        nota = 7.0
    elif fraccion < 0.60:
        nota = 1.0 + (fraccion / 0.60) * 3.0
    else:
        nota = 4.0 + ((fraccion - 0.60) / 0.40) * 3.0

    nota = round(nota, 1)

    return {
        'correctas':   correctas,
        'incorrectas': incorrectas,
        'omitidas':    omitidas,
        'total':       total_preguntas,
        'nota':        nota,
        'porcentaje':  round(fraccion * 100, 1),
        'detalle':     detalle,
    }
