from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import json, os, uuid, shutil, tempfile

from omr import procesar_hoja, calcular_nota

app = FastAPI(title="ServidorOMR - Calco")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Almacenamiento de pautas en JSON ─────────────────────────────────────────
PAUTAS_FILE = "pautas.json"

def cargar_pautas() -> dict:
    if os.path.exists(PAUTAS_FILE):
        with open(PAUTAS_FILE, "r") as f:
            return json.load(f)
    return {}

def guardar_pautas(pautas: dict):
    with open(PAUTAS_FILE, "w") as f:
        json.dump(pautas, f, ensure_ascii=False, indent=2)


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
def crear_pauta(pauta: Pauta):
    pautas = cargar_pautas()
    pid = str(uuid.uuid4())[:8]
    pautas[pid] = {
        "id": pid,
        "nombre": pauta.nombre,
        "asignatura": pauta.asignatura,
        "respuestas": pauta.respuestas,
        "total_preguntas": pauta.total_preguntas,
    }
    guardar_pautas(pautas)
    return {"ok": True, "id": pid, "nombre": pauta.nombre}


@app.get("/pautas")
def listar_pautas():
    pautas = cargar_pautas()
    lista = [{"id": v["id"], "nombre": v["nombre"], "asignatura": v.get("asignatura",""),
              "total_preguntas": v.get("total_preguntas", 80)} for v in pautas.values()]
    return {"total": len(lista), "pautas": lista}


@app.get("/pautas/{pauta_id}")
def obtener_pauta(pauta_id: str):
    pautas = cargar_pautas()
    if pauta_id not in pautas:
        raise HTTPException(status_code=404, detail="Pauta no encontrada")
    return pautas[pauta_id]


@app.delete("/pautas/{pauta_id}")
def eliminar_pauta(pauta_id: str):
    pautas = cargar_pautas()
    if pauta_id not in pautas:
        raise HTTPException(status_code=404, detail="Pauta no encontrada")
    del pautas[pauta_id]
    guardar_pautas(pautas)
    return {"ok": True}


# Procesamiento OMR
@app.post("/procesar")
async def procesar(
    imagen: UploadFile = File(...),
    pauta_id: str = "",
    nombre_estudiante: str = "",
    rut: str = "",
):
    pautas = cargar_pautas()
    if pauta_id not in pautas:
        raise HTTPException(status_code=404, detail=f"Pauta '{pauta_id}' no encontrada")

    pauta = pautas[pauta_id]

    # Guardar imagen temporal
    suffix = os.path.splitext(imagen.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(imagen.file, tmp)
        ruta_tmp = tmp.name

    try:
        respuestas = procesar_hoja(ruta_tmp)
        resultado  = calcular_nota(respuestas, pauta["respuestas"], pauta.get("total_preguntas", 80))
    finally:
        os.unlink(ruta_tmp)

    return {
        "ok": True,
        "estudiante": nombre_estudiante,
        "rut": rut,
        "pauta": pauta["nombre"],
        **resultado,
    }


# Calibración (devuelve imagen con burbujas detectadas)
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


# Archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
