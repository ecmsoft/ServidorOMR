from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, List
import os, shutil, tempfile, io
import cv2
import numpy as np

import db
from omr import procesar_hoja, procesar_hoja_imagen, generar_pdf_procesado, calcular_nota, corregir_perspectiva, leer_imagen, binarizar, OPC_X, Y_INICIO, Y_FIN, N_FILAS, RADIO_BURBUJA, ALTO, ANCHO, UMBRAL_MARCADO, UMBRAL_PARCIAL, leer_burbuja, generar_txt

app = FastAPI(title="ServidorOMR - Calco")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    db.init_db()


# ── Modelos ───────────────────────────────────────────────────────────────────
class Pauta(BaseModel):
    nombre: str                        # ej. "Ensayo LG-1 Mayo 2025"
    asignatura: Optional[str] = ""     # ej. "Lenguaje"
    respuestas: dict                   # {"1": "A", "2": "C", ...}
    total_preguntas: Optional[int] = 80


class ResultadoOMR(BaseModel):
    pauta_id: str
    nombre_estudiante: Optional[str] = ""
    rut: Optional[str] = ""


# ── Rutas ─────────────────────────────────────────────────────────────────────

@app.get("/estado")
def estado():
    return {"ok": True, "mensaje": "✅ ServidorOMR funcionando"}


# Pautas CRUD
@app.post("/pautas")
def crear_pauta_endpoint(pauta: Pauta):
    pid = db.crear_pauta(pauta.nombre, pauta.asignatura or "", pauta.respuestas, pauta.total_preguntas)
    return {"ok": True, "id": pid, "nombre": pauta.nombre}


@app.get("/pautas")
def listar_pautas_endpoint():
    lista = db.listar_pautas()
    return {"total": len(lista), "pautas": lista}


@app.get("/pautas/{pauta_id}")
def obtener_pauta_endpoint(pauta_id: str):
    pauta = db.obtener_pauta(pauta_id)
    if pauta is None:
        raise HTTPException(status_code=404, detail="Pauta no encontrada")
    return pauta


@app.delete("/pautas/{pauta_id}")
def eliminar_pauta_endpoint(pauta_id: str):
    if not db.eliminar_pauta(pauta_id):
        raise HTTPException(status_code=404, detail="Pauta no encontrada")
    return {"ok": True}


# Resultados
@app.get("/resultados")
def listar_resultados_endpoint(pauta_id: str = "", curso: str = ""):
    lista = db.listar_resultados(pauta_id or None, curso or None)
    return {"total": len(lista), "resultados": lista}


@app.get("/resultados/{resultado_id}/txt")
def descargar_resultado_txt(resultado_id: int):
    resultado = db.obtener_resultado(resultado_id)
    if resultado is None:
        raise HTTPException(status_code=404, detail="Resultado no encontrado")
    nombre_archivo = f"{resultado['nombre_estudiante'] or 'alumno'}_{resultado['pauta_nombre'] or ''}.txt".replace(" ", "_")
    return PlainTextResponse(
        resultado["txt"] or "",
        headers={"Content-Disposition": f'attachment; filename="{nombre_archivo}"'},
    )


# Procesamiento OMR
@app.post("/procesar")
async def procesar(
    imagen: UploadFile = File(...),
    pauta_id: str = Form(""),
    nombre_estudiante: str = Form(""),
    curso: str = Form(""),
    rut: str = Form(""),
    fecha_entrega: str = Form(""),
):
    pauta_id = pauta_id.strip()
    pauta = db.obtener_pauta(pauta_id)
    if pauta is None:
        raise HTTPException(status_code=404, detail=f"Pauta '{pauta_id}' no encontrada.")

    # Guardar imagen temporal
    suffix = os.path.splitext(imagen.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(imagen.file, tmp)
        ruta_tmp = tmp.name

    try:
        img = leer_imagen(ruta_tmp)
        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen.")
        respuestas, ambiguas, img_norm = procesar_hoja_imagen(img)
        total      = pauta.get("total_preguntas", 80)
        resultado  = calcular_nota(respuestas, pauta["respuestas"], total, ambiguas=ambiguas)
        imagen_pdf = generar_pdf_procesado(img_norm)
    finally:
        os.unlink(ruta_tmp)

    txt = generar_txt(nombre_estudiante, curso, pauta["nombre"], respuestas, total,
                       rut=rut, fecha_entrega=fecha_entrega)

    resultado_id = db.guardar_resultado(
        pauta_id=pauta_id,
        nombre_estudiante=nombre_estudiante,
        rut=rut,
        curso=curso,
        fecha_entrega=fecha_entrega,
        respuestas={str(k): v for k, v in respuestas.items()},
        detalle=resultado["detalle"],
        correctas=resultado["correctas"],
        incorrectas=resultado["incorrectas"],
        omitidas=resultado["omitidas"],
        nota=resultado["nota"],
        porcentaje=resultado["porcentaje"],
        txt=txt,
        imagen_pdf=imagen_pdf,
        ambiguas=ambiguas,
    )

    return {
        "ok": True,
        "id": resultado_id,
        "estudiante": nombre_estudiante,
        "curso": curso,
        "rut": rut,
        "fecha_entrega": fecha_entrega,
        "pauta": pauta["nombre"],
        "txt": txt,
        "ambiguas_detalle": ambiguas,
        "requiere_confirmacion": bool(ambiguas),
        **resultado,
    }


class ConfirmacionRespuestas(BaseModel):
    # {"3": "B", "47": null (= "marca multiple / no estoy seguro")}
    respuestas: dict


@app.post("/resultados/{resultado_id}/confirmar")
def confirmar_resultado(resultado_id: int, body: ConfirmacionRespuestas):
    """
    Aplica las respuestas confirmadas por el usuario sobre las preguntas que
    quedaron ambiguas (marca parcial o doble marca) y recalcula la nota.
    Una pregunta ambigua sin una opción confirmada (valor null) queda
    registrada como "marca múltiple / no estoy seguro" y se mantiene en la
    categoría 'ambigua' del detalle, sin puntuar.
    """
    resultado_db = db.obtener_resultado(resultado_id)
    if resultado_db is None:
        raise HTTPException(status_code=404, detail="Resultado no encontrado")

    pauta = db.obtener_pauta(resultado_db["pauta_id"])
    if pauta is None:
        raise HTTPException(status_code=404, detail="Pauta del resultado no encontrada")

    respuestas = {int(k): v for k, v in (resultado_db["respuestas"] or {}).items()}
    ambiguas_originales = resultado_db["ambiguas"] or {}

    ambiguas_restantes = {}
    for q_str, opciones in ambiguas_originales.items():
        q = int(q_str)
        if q_str in body.respuestas:
            elegida = body.respuestas[q_str]
            if elegida:
                respuestas[q] = elegida
            else:
                ambiguas_restantes[q] = opciones   # marca múltiple / no está seguro
        else:
            ambiguas_restantes[q] = opciones       # no vino en la confirmación, sigue pendiente

    total = pauta.get("total_preguntas", 80)
    resultado = calcular_nota(respuestas, pauta["respuestas"], total, ambiguas=ambiguas_restantes)
    txt = generar_txt(resultado_db["nombre_estudiante"], resultado_db["curso"], pauta["nombre"],
                       respuestas, total, rut=resultado_db["rut"],
                       fecha_entrega=str(resultado_db["fecha_entrega"] or ""))

    db.confirmar_resultado(
        resultado_id=resultado_id,
        respuestas={str(k): v for k, v in respuestas.items()},
        detalle=resultado["detalle"],
        correctas=resultado["correctas"],
        incorrectas=resultado["incorrectas"],
        omitidas=resultado["omitidas"],
        nota=resultado["nota"],
        porcentaje=resultado["porcentaje"],
        txt=txt,
        ambiguas=ambiguas_restantes,
    )

    return {
        "ok": True,
        "id": resultado_id,
        "estudiante": resultado_db["nombre_estudiante"],
        "curso": resultado_db["curso"],
        "pauta": pauta["nombre"],
        "txt": txt,
        "ambiguas_detalle": ambiguas_restantes,
        "requiere_confirmacion": bool(ambiguas_restantes),
        **resultado,
    }


# Calibración JSON
@app.post("/calibrar")
async def calibrar(imagen: UploadFile = File(...)):
    suffix = os.path.splitext(imagen.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(imagen.file, tmp)
        ruta_tmp = tmp.name
    try:
        respuestas = procesar_hoja(ruta_tmp)
        detectadas = sum(1 for v in respuestas.values() if v is not None)
        return {"ok": True, "detectadas": detectadas, "respuestas": respuestas}
    finally:
        os.unlink(ruta_tmp)


# Debug visual — devuelve imagen con burbujas marcadas
@app.post("/debug-imagen")
async def debug_imagen(imagen: UploadFile = File(...)):
    """
    Recibe una foto de hoja, aplica corrección de perspectiva y devuelve
    la imagen procesada con:
      - Círculo VERDE: burbuja detectada como marcada
      - Círculo ROJO:  posición esperada (no marcada)
    """
    suffix = os.path.splitext(imagen.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(imagen.file, tmp)
        ruta_tmp = tmp.name

    try:
        try:
            img = leer_imagen(ruta_tmp)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"No se pudo leer la imagen: {e}")
        if img is None:
            raise HTTPException(status_code=400, detail="Formato de imagen no soportado")

        # Corregir perspectiva
        img_norm = corregir_perspectiva(img)

        # Binarizar (misma función que usa /procesar — no duplicar el criterio
        # de "marcada" para que este debug siempre refleje el resultado real)
        binaria = binarizar(img_norm)

        # Convertir binarizada a BGR para dibujar encima en color
        visual = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

        OPCIONES = ['A', 'B', 'C', 'D', 'E']

        # Dibujar posiciones de burbujas
        for col_idx, xs in enumerate(OPC_X):
            offset_q = col_idx * N_FILAS
            for fila in range(N_FILAS):
                num_q = fila + 1 + offset_q
                y_rel = Y_INICIO + (Y_FIN - Y_INICIO) * fila / (N_FILAS - 1)
                cy = int(y_rel * ALTO)

                for i, x_rel in enumerate(xs):
                    cx = int(x_rel * ANCHO)
                    fraccion = leer_burbuja(binaria, cx, cy)
                    marcada = fraccion >= UMBRAL_MARCADO
                    parcial = (not marcada) and fraccion >= UMBRAL_PARCIAL

                    if marcada:
                        color, grosor = (0, 200, 0), -1        # verde relleno = marcada
                    elif parcial:
                        color, grosor = (0, 210, 255), 3        # amarillo borde grueso = grafito parcial (no llega al umbral)
                    else:
                        color, grosor = (0, 0, 220), 2          # rojo borde = vacía

                    cv2.circle(visual, (cx, cy), RADIO_BURBUJA, color, grosor)

                    # Fracción de relleno sobre la burbuja (para calibración)
                    txt = f"{fraccion:.2f}"
                    pos = (cx - 12, cy - RADIO_BURBUJA - 4)
                    cv2.putText(visual, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 3)      # contorno
                    cv2.putText(visual, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1) # relleno

                    # Número de pregunta en la primera burbuja de cada fila
                    if i == 0:
                        cv2.putText(visual, str(num_q),
                                    (cx - RADIO_BURBUJA - 20, cy + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        # ── Overlay: bloque de texto con respuestas detectadas ────────────
        OPCIONES = ['A', 'B', 'C', 'D', 'E']
        _LETRA_A_NUM = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}

        # Reconstruir respuestas desde el visual ya dibujado
        respuestas_debug = {}
        for col_idx, xs in enumerate(OPC_X):
            offset_q = col_idx * N_FILAS
            for fila in range(N_FILAS):
                num_q  = fila + 1 + offset_q
                y_rel  = Y_INICIO + (Y_FIN - Y_INICIO) * fila / (N_FILAS - 1)
                cy     = int(y_rel * ALTO)
                marcadas = [OPCIONES[i] for i, x_rel in enumerate(xs)
                            if leer_burbuja(binaria, int(x_rel * ANCHO), cy) >= UMBRAL_MARCADO]
                respuestas_debug[num_q] = marcadas[0] if len(marcadas) == 1 else (marcadas[0] if marcadas else None)

        # Panel lateral derecho con el listado
        panel_w = 220
        panel = np.ones((ALTO, panel_w, 3), dtype=np.uint8) * 240
        font  = cv2.FONT_HERSHEY_SIMPLEX
        y_txt = 20
        cv2.putText(panel, "RESPUESTAS", (10, y_txt), font, 0.5, (0, 0, 0), 1)
        y_txt += 20
        for q in range(1, 81):
            letra = respuestas_debug.get(q)
            num   = _LETRA_A_NUM.get(letra, 0)
            texto = f"P{q:02d}: {letra or '-'} ({num})"
            color = (0, 120, 0) if letra else (150, 150, 150)
            cv2.putText(panel, texto, (10, y_txt), font, 0.38, color, 1)
            y_txt += 21
            if y_txt > ALTO - 10:
                break

        visual_con_panel = np.hstack([visual, panel])

        # Escalar a la mitad para que no sea tan pesado
        h, w = visual_con_panel.shape[:2]
        visual_small = cv2.resize(visual_con_panel, (w // 2, h // 2))

        _, buf = cv2.imencode('.jpg', visual_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")
    finally:
        os.unlink(ruta_tmp)


# Panel admin (profesores)
@app.get("/", response_class=HTMLResponse)
def panel():
    ruta_html = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(ruta_html, "r", encoding="utf-8") as f:
        return f.read()


# Portal alumnos (web mobile)
@app.get("/portal", response_class=HTMLResponse)
def portal():
    ruta_html = os.path.join(os.path.dirname(__file__), "static", "portal.html")
    with open(ruta_html, "r", encoding="utf-8") as f:
        return f.read()


# Hojas de respuesta imprimibles
@app.get("/hoja/80", response_class=HTMLResponse)
def hoja_80():
    ruta_html = os.path.join(os.path.dirname(__file__), "static", "hoja_80.html")
    with open(ruta_html, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/hoja/65", response_class=HTMLResponse)
def hoja_65():
    ruta_html = os.path.join(os.path.dirname(__file__), "static", "hoja_65.html")
    with open(ruta_html, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/hoja/preuc", response_class=HTMLResponse)
def hoja_preuc():
    ruta_html = os.path.join(os.path.dirname(__file__), "static", "hoja_preuc.html")
    with open(ruta_html, "r", encoding="utf-8") as f:
        return f.read()


# Archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
