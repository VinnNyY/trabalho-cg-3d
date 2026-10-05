"""
PARTE 2 — Transformações geométricas 3D em coordenadas homogêneas (4x4).

Convenções adotadas em todo o projeto
-------------------------------------
* Vetores são COLUNA:  p' = M @ p,  com p = (x, y, z, 1)^T.
* As matrizes NumPy são guardadas em ordem "linha-maior" (row-major), exatamente
  como escritas no quadro.  O OpenGL espera "coluna-maior" (column-major), por
  isso, ao enviar para o pipeline usamos ``to_gl(M)`` (= M transposta).
* Ângulos em GRAUS na interface pública (como gluPerspective/glRotate).

Ordem de composição
-------------------
    M = T · R · S
Como o vetor é multiplicado à direita, a matriz MAIS À DIREITA age PRIMEIRO:
o objeto é escalado na origem, depois girado em torno da origem e só então
transladado. Se a ordem fosse T aplicado antes de R, a rotação ocorreria em
torno da origem do mundo e "arrastaria" o objeto em órbita (deslocamento
indesejado) — veja ``tests/test_transforms.py::test_ordem_importa``.
"""
from __future__ import annotations

import math

import numpy as np

Mat4 = np.ndarray


# --------------------------------------------------------------------------- #
# Matrizes elementares
# --------------------------------------------------------------------------- #
def identity() -> Mat4:
    return np.eye(4, dtype=np.float64)


def translation(tx: float, ty: float, tz: float) -> Mat4:
    """T(tx, ty, tz)."""
    return np.array(
        [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def scale(sx: float, sy: float, sz: float) -> Mat4:
    """S(sx, sy, sz)."""
    return np.array(
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rotation_x(theta_deg: float) -> Mat4:
    """R_x(θ): rotação anti-horária em torno do eixo X (regra da mão direita)."""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rotation_y(theta_deg: float) -> Mat4:
    """R_y(θ). Observe o sinal invertido do seno em relação a R_x e R_z
    (consequência da ordem cíclica z → x)."""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rotation_z(theta_deg: float) -> Mat4:
    """R_z(θ)."""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# --------------------------------------------------------------------------- #
# Composição
# --------------------------------------------------------------------------- #
def compose(*matrices: Mat4) -> Mat4:
    """Produto da esquerda para a direita: compose(A, B, C) = A·B·C
    (C é aplicada primeiro ao vértice)."""
    result = identity()
    for m in matrices:
        result = result @ m
    return result


def model_matrix(t: Mat4, r: Mat4, s: Mat4) -> Mat4:
    """Matriz de Modelo  M = T · R · S  (Espaço do Objeto → Espaço do Mundo)."""
    return t @ r @ s


def transform_point(m: Mat4, p) -> np.ndarray:
    """Aplica M a um ponto 3D (w = 1) e devolve o ponto cartesiano."""
    ph = m @ np.array([p[0], p[1], p[2], 1.0])
    return ph[:3] / ph[3]


def transform_direction(m: Mat4, v) -> np.ndarray:
    """Aplica M a uma direção (w = 0): a translação não a afeta."""
    return (m @ np.array([v[0], v[1], v[2], 0.0]))[:3]


def normal_matrix(m: Mat4) -> np.ndarray:
    """(M_3x3^-1)^T — matriz correta para transformar normais quando há
    escala não uniforme."""
    return np.linalg.inv(m[:3, :3]).T


# --------------------------------------------------------------------------- #
# Câmera (Matriz de Visão) e Projeção
# --------------------------------------------------------------------------- #
def look_at(eye, center, up) -> Mat4:
    """Matriz de Visão sintética, equivalente a gluLookAt.

    Constrói a base ortonormal da câmera:
        f = normalize(center - eye)     (para onde a câmera olha)
        s = normalize(f × up)           (direita)
        u = s × f                       (cima verdadeiro)
    e a inversa da transformação câmera→mundo:  V = R^T · T(-eye).
    """
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)

    rot = identity()
    rot[0, :3] = s
    rot[1, :3] = u
    rot[2, :3] = -f
    return rot @ translation(-eye[0], -eye[1], -eye[2])


def frustum(left, right, bottom, top, near, far) -> Mat4:
    """Projeção em perspectiva genérica, equivalente a glFrustum."""
    if near <= 0 or far <= near:
        raise ValueError("Requer 0 < near < far")
    return np.array(
        [
            [2 * near / (right - left), 0.0, (right + left) / (right - left), 0.0],
            [0.0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> Mat4:
    """Projeção em perspectiva simétrica, equivalente a gluPerspective.
    É um caso particular de frustum com top = near·tan(fovy/2)."""
    top = near * math.tan(math.radians(fovy_deg) / 2.0)
    right = top * aspect
    return frustum(-right, right, -top, top, near, far)


# --------------------------------------------------------------------------- #
# Ponte com o OpenGL
# --------------------------------------------------------------------------- #
def to_gl(m: Mat4) -> np.ndarray:
    """Converte para o layout coluna-maior (float32) aceito por glLoadMatrixf."""
    return np.ascontiguousarray(m.T, dtype=np.float32)
