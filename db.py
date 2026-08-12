"""
db.py — Acceso a Postgres para pautas y resultados.
Reemplaza el almacenamiento anterior en pautas.json (filesystem efímero).
"""

import os
import uuid
from contextlib import contextmanager

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor, Json

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        dsn = os.environ["DATABASE_URL"]
        _pool = psycopg2.pool.SimpleConnectionPool(1, 10, dsn)
    return _pool


@contextmanager
def _conn():
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def init_db():
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pautas (
                id TEXT PRIMARY KEY,
                nombre TEXT NOT NULL,
                asignatura TEXT,
                total_preguntas INTEGER NOT NULL DEFAULT 80,
                respuestas JSONB NOT NULL,
                creado_en TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS resultados (
                id SERIAL PRIMARY KEY,
                pauta_id TEXT REFERENCES pautas(id),
                nombre_estudiante TEXT,
                rut TEXT,
                curso TEXT,
                fecha_entrega DATE,
                respuestas JSONB,
                detalle JSONB,
                correctas INTEGER,
                incorrectas INTEGER,
                omitidas INTEGER,
                nota NUMERIC(3,1),
                porcentaje NUMERIC(5,1),
                txt TEXT,
                creado_en TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)


# ── Pautas ────────────────────────────────────────────────────────────────────

def crear_pauta(nombre: str, asignatura: str, respuestas: dict, total_preguntas: int) -> str:
    pid = str(uuid.uuid4())[:8]
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pautas (id, nombre, asignatura, total_preguntas, respuestas)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (pid, nombre, asignatura, total_preguntas, Json(respuestas)),
        )
    return pid


def listar_pautas() -> list:
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, nombre, asignatura, total_preguntas
            FROM pautas ORDER BY creado_en DESC
        """)
        return cur.fetchall()


def obtener_pauta(pauta_id: str):
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, nombre, asignatura, total_preguntas, respuestas
            FROM pautas WHERE id = %s
        """, (pauta_id,))
        return cur.fetchone()


def eliminar_pauta(pauta_id: str) -> bool:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM pautas WHERE id = %s", (pauta_id,))
        return cur.rowcount > 0


# ── Resultados ────────────────────────────────────────────────────────────────

def guardar_resultado(pauta_id: str, nombre_estudiante: str, rut: str, curso: str,
                       fecha_entrega: str, respuestas: dict, detalle: dict,
                       correctas: int, incorrectas: int, omitidas: int,
                       nota: float, porcentaje: float, txt: str) -> int:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO resultados (
                pauta_id, nombre_estudiante, rut, curso, fecha_entrega,
                respuestas, detalle, correctas, incorrectas, omitidas,
                nota, porcentaje, txt
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                pauta_id, nombre_estudiante, rut, curso, fecha_entrega or None,
                Json(respuestas), Json(detalle), correctas, incorrectas, omitidas,
                nota, porcentaje, txt,
            ),
        )
        return cur.fetchone()[0]


def listar_resultados(pauta_id: str = None, curso: str = None) -> list:
    filtros = []
    valores = []
    if pauta_id:
        filtros.append("r.pauta_id = %s")
        valores.append(pauta_id)
    if curso:
        filtros.append("r.curso = %s")
        valores.append(curso)
    where = ("WHERE " + " AND ".join(filtros)) if filtros else ""

    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT r.id, r.pauta_id, p.nombre AS pauta_nombre, r.nombre_estudiante,
                   r.rut, r.curso, r.fecha_entrega, r.correctas, r.incorrectas,
                   r.omitidas, r.nota, r.porcentaje, r.creado_en
            FROM resultados r
            LEFT JOIN pautas p ON p.id = r.pauta_id
            {where}
            ORDER BY r.creado_en DESC
        """, valores)
        return cur.fetchall()


def obtener_resultado(resultado_id: int):
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT r.*, p.nombre AS pauta_nombre
            FROM resultados r
            LEFT JOIN pautas p ON p.id = r.pauta_id
            WHERE r.id = %s
        """, (resultado_id,))
        return cur.fetchone()
