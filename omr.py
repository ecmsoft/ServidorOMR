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
    Convierte un archivo HEIC a JPEG usando heif-convert (libheif-examples).
    Retorna la ruta del archivo JPEG temporal, o None si falla.
    """
    ruta_jpg = ruta_heic + '_converted.jpg'
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
RADIO_BURBUJA  = 10    # px en imagen normalizada (interior de la burbuja)
UMBRAL_MARCADO = 0.50  # fracción de píxeles oscuros para considerar marcada

OPCIONES = ['A', 'B', 'C', 'D', 'E']


# ── Utilidad: ordenar 4 puntos en [TL, TR, BL, BR] ───────────────────────────
def _ordenar_esquinas(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s    = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.array([tl, tr, bl, br], dtype=np.float32)


# ── Método 1: marcadores cuadrados negros en esquinas ────────────────────────
def _detectar_marcadores(gris: np.ndarray):
    """
    Busca los 4 cuadrados negros de las esquinas.
    Retorna [TL, TR, BL, BR] o None.
    """
    h, w = gris.shape
    # Umbral adaptativo (Otsu) en vez de uno fijo: el brillo real de una foto
    # de celular varía demasiado (poca luz, sombras) para un valor constante.
    gris_suave = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(gris_suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for c in contornos:
        area = cv2.contourArea(c)
        min_a = (w * h) * 0.0003   # al menos 0.03% del área
        max_a = (w * h) * 0.015    # máximo 1.5%
        if min_a < area < max_a:
            x, y, cw, ch = cv2.boundingRect(c)
            aspecto = cw / ch if ch > 0 else 0
            if 0.5 < aspecto < 2.0:
                candidatos.append((x + cw // 2, y + ch // 2))

    if len(candidatos) < 4:
        return None

    # Tomar los 4 más cercanos a cada esquina
    esquinas_img = [
        (0,   0  ),   # TL
        (w,   0  ),   # TR
        (0,   h  ),   # BL
        (w,   h  ),   # BR
    ]
    resultado = []
    usados = set()
    for ex, ey in esquinas_img:
        mejor = min(
            (i for i in range(len(candidatos)) if i not in usados),
            key=lambda i: (candidatos[i][0]-ex)**2 + (candidatos[i][1]-ey)**2,
            default=None
        )
        if mejor is None:
            return None
        usados.add(mejor)
        resultado.append(candidatos[mejor])

    tl, tr, bl, br = resultado
    return np.array([tl, tr, bl, br], dtype=np.float32)


# ── Método 2: contorno de la hoja blanca contra fondo oscuro ─────────────────
def _detectar_hoja_blanca(imagen: np.ndarray):
    """
    Detecta la hoja como el mayor contorno claro de la imagen.
    Retorna [TL, TR, BL, BR] o None.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen.copy()
    h, w = gris.shape

    # Umbral adaptativo (Otsu) en vez de uno fijo, por la misma razón que en
    # _detectar_marcadores: la iluminación real de estas fotos es muy variable.
    gris_suave = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(gris_suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel2)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    c = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * h * w:
        return None

    peri   = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        return _ordenar_esquinas(approx)

    rect = cv2.minAreaRect(c)
    box  = cv2.boxPoints(rect)
    return _ordenar_esquinas(box)


# ── Corrección de perspectiva (combinada) ─────────────────────────────────────
# Los marcadores están a 4mm del borde, son 8×8mm → centro a 8mm del borde
_MX = 8 / 215.9 * ANCHO   # ~46 px
_MY = 8 / 279.4 * ALTO    # ~50 px

# Posición CONOCIDA de los centros de marcadores en imagen normalizada
_DST_MARCADORES = np.array([
    [_MX,        _MY       ],   # TL
    [ANCHO - _MX, _MY      ],   # TR
    [_MX,        ALTO - _MY],   # BL
    [ANCHO - _MX, ALTO - _MY],  # BR
], dtype=np.float32)

_DST_FULL = np.array([
    [0,     0    ],
    [ANCHO, 0    ],
    [0,     ALTO ],
    [ANCHO, ALTO ],
], dtype=np.float32)


def corregir_perspectiva(imagen: np.ndarray) -> np.ndarray:
    """
    1. Warp grueso detectando contorno de hoja blanca
    2. Refinamiento preciso usando los 4 marcadores de esquina,
       mapeándolos a su posición CONOCIDA (no a las esquinas del canvas)
    """
    # ── Paso 1: detectar y warpear la hoja blanca ──────────────────────────
    esquinas_hoja = _detectar_hoja_blanca(imagen)
    if esquinas_hoja is None:
        img_warp = cv2.resize(imagen, (ANCHO, ALTO))
    else:
        M1 = cv2.getPerspectiveTransform(esquinas_hoja, _DST_FULL)
        img_warp = cv2.warpPerspective(imagen, M1, (ANCHO, ALTO))

    # ── Paso 2: refinar con marcadores (si se detectan) ───────────────────
    gris_warp = cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY) if len(img_warp.shape) == 3 else img_warp.copy()
    marcadores = _detectar_marcadores(gris_warp)
    if marcadores is not None:
        # Mapear centros detectados → posición conocida en coordenadas normalizadas
        M2 = cv2.getPerspectiveTransform(marcadores, _DST_MARCADORES)
        img_warp = cv2.warpPerspective(img_warp, M2, (ANCHO, ALTO))

    return img_warp


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

    # 2. Corregir iluminación despareja (sombras de la foto) antes de binarizar:
    #    dividir por una versión muy difuminada de si misma aplana el gradiente
    #    de sombra sin perder las marcas de lápiz (mucho más locales/finas).
    gris  = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY) if len(img_norm.shape) == 3 else img_norm
    fondo = cv2.GaussianBlur(gris, (0, 0), sigmaX=31)
    plano = cv2.divide(gris, fondo, scale=255)
    _, binaria = cv2.threshold(plano, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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


# ── Generación de archivo TXT ─────────────────────────────────────────────────
_LETRA_A_NUM = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}

def generar_txt(nombre: str, curso: str, nombre_pauta: str,
                respuestas: dict, total: int = 80,
                rut: str = "", fecha_entrega: str = "") -> str:
    """
    Genera el contenido del archivo TXT de resultados.
    Respuestas codificadas: A=1, B=2, C=3, D=4, E=5, omitida=0
    """
    nums = [str(_LETRA_A_NUM.get(respuestas.get(q), 0)) for q in range(1, total + 1)]
    lineas = [
        f"NOMBRE: {nombre}",
        f"RUT: {rut}",
        f"CURSO: {curso}",
        f"FECHA_ENTREGA: {fecha_entrega}",
        f"PAUTA: {nombre_pauta}",
        f"RESPUESTAS: {','.join(nums)}",
    ]
    return '\n'.join(lineas)


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
