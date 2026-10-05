"""
PARTE 1 e PARTE 3 — Modelagem geométrica por malha poligonal + vetores normais.

Cada sólido é definido por:
* ``vertices``: lista de pontos (x, y, z) no Espaço do Objeto, centrados na
  origem para que rotação e escala aconteçam em torno do próprio centro.
* ``faces``: lista de índices para ``vertices``. Todas as faces seguem a
  ordem ANTI-HORÁRIA quando vistas DE FORA do sólido. Isso garante que
  N = (V1 − V0) × (V2 − V0) aponte para fora (regra da mão direita) e
  permite usar GL_CULL_FACE com segurança.

Faces de 3 vértices → GL_TRIANGLES.  Faces de 4 vértices → GL_QUADS
(a base da pirâmide usa um quad, as laterais usam triângulos).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    name: str
    vertices: np.ndarray            # (V, 3) float64
    faces: list[tuple[int, ...]]    # índices por face
    face_normals: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.face_normals = compute_face_normals(self.vertices, self.faces)

    @property
    def triangles(self) -> list[int]:
        return [i for i, f in enumerate(self.faces) if len(f) == 3]

    @property
    def quads(self) -> list[int]:
        return [i for i, f in enumerate(self.faces) if len(f) == 4]

    def face_centroid(self, i: int) -> np.ndarray:
        return self.vertices[list(self.faces[i])].mean(axis=0)

    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    def euler_characteristic(self) -> int:
        """V − A + F (deve ser 2 para poliedros convexos fechados)."""
        edges = set()
        for f in self.faces:
            for a, b in zip(f, f[1:] + f[:1]):
                edges.add((min(a, b), max(a, b)))
        return len(self.vertices) - len(edges) + len(self.faces)


# --------------------------------------------------------------------------- #
# Vetores normais
# --------------------------------------------------------------------------- #
def face_normal(v0, v1, v2) -> np.ndarray:
    """N = (V1 − V0) × (V2 − V0), normalizado para |N| = 1."""
    v0, v1, v2 = (np.asarray(v, dtype=np.float64) for v in (v0, v1, v2))
    n = np.cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(n)
    if length < 1e-12:
        raise ValueError("Face degenerada: vértices colineares")
    return n / length


def compute_face_normals(vertices: np.ndarray, faces) -> np.ndarray:
    """Uma normal por face. Para quads usa os três primeiros vértices
    (as faces são planas, então qualquer trio não colinear serve)."""
    return np.array([face_normal(*(vertices[i] for i in f[:3])) for f in faces])


# --------------------------------------------------------------------------- #
# Sólidos
# --------------------------------------------------------------------------- #
def pyramid(base: float = 1.6, height: float = 1.6) -> Mesh:
    """Pirâmide de base quadrada. Centroide de volume na origem
    (para pirâmide, fica a 1/4 da altura a partir da base)."""
    h = base / 2.0
    y_base = -height / 4.0
    y_apex = 3.0 * height / 4.0
    vertices = [
        (-h, y_base, h),    # 0 frente-esquerda
        (h, y_base, h),     # 1 frente-direita
        (h, y_base, -h),    # 2 trás-direita
        (-h, y_base, -h),   # 3 trás-esquerda
        (0.0, y_apex, 0.0), # 4 ápice
    ]
    faces = [
        (0, 1, 4),          # frente
        (1, 2, 4),          # direita
        (2, 3, 4),          # trás
        (3, 0, 4),          # esquerda
        (0, 3, 2, 1),       # base (quad, vista de baixo em sentido anti-horário)
    ]
    return Mesh("Pirâmide", np.array(vertices), faces)


def octahedron(r: float = 1.2) -> Mesh:
    vertices = [
        (r, 0, 0), (-r, 0, 0),
        (0, r, 0), (0, -r, 0),
        (0, 0, r), (0, 0, -r),
    ]
    # 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z
    faces = [
        (4, 0, 2), (0, 5, 2), (5, 1, 2), (1, 4, 2),   # metade superior
        (0, 4, 3), (5, 0, 3), (1, 5, 3), (4, 1, 3),   # metade inferior
    ]
    return Mesh("Octaedro", np.array(vertices, dtype=np.float64), faces)


def tetrahedron(r: float = 1.3) -> Mesh:
    """Tetraedro regular inscrito numa esfera de raio r (vértices alternados
    de um cubo)."""
    k = r / math.sqrt(3.0)
    vertices = [
        (k, k, k),
        (-k, -k, k),
        (-k, k, -k),
        (k, -k, -k),
    ]
    faces = [
        (0, 2, 1),
        (0, 1, 3),
        (0, 3, 2),
        (1, 2, 3),
    ]
    return Mesh("Tetraedro", np.array(vertices), faces)


def all_meshes() -> list[Mesh]:
    return [pyramid(), octahedron(), tetrahedron()]
