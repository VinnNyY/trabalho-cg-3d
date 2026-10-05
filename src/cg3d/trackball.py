"""
PARTE 4 — Trackball Virtual (Shoemake / Bell).

Fluxo de um arraste do mouse:

    1. clique  em (x0, y0)  →  p0 = project_to_sphere(x0, y0)
    2. arraste até (x1, y1) →  p1 = project_to_sphere(x1, y1)
    3. eixo  = p0 × p1                         (produto vetorial)
       θ     = acos( p0 · p1 / (|p0||p1|) )    (produto escalar)
    4. q_drag  = Quaternion.from_axis_angle(eixo, θ · sensibilidade)
    5. q_total = q_drag * q_total              (acumula a orientação)
    6. R = q_total.to_matrix()  → enviada ao OpenGL a cada quadro.

Como a rotação nunca passa por ângulos de Euler, não existe Gimbal Lock.
"""
from __future__ import annotations

import math

import numpy as np

from .quaternion import Quaternion


def screen_to_ndc(x: float, y: float, width: int, height: int):
    """Pixels da janela (origem no canto superior esquerdo, y para baixo)
    → coordenadas normalizadas [-1, 1] com y para cima.
    Usa o menor lado da janela como raio para a esfera ficar redonda
    mesmo com janelas retangulares."""
    radius = min(width, height) / 2.0
    nx = (x - width / 2.0) / radius
    ny = (height / 2.0 - y) / radius
    return nx, ny


def project_to_sphere(nx: float, ny: float) -> np.ndarray:
    """Mapeia (x, y) ∈ [-1, 1]² para um vetor unitário no hemisfério virtual
    de raio 1 voltado para o observador (z ≥ 0).

    * Dentro do círculo (x² + y² ≤ 1):  z = sqrt(1 − x² − y²)
    * Fora do círculo: o ponto é levado à borda do hemisfério (z = 0),
      o que produz rotação em torno do eixo de visão (Z).
    """
    d2 = nx * nx + ny * ny
    if d2 <= 1.0:
        return np.array([nx, ny, math.sqrt(1.0 - d2)])
    d = math.sqrt(d2)
    return np.array([nx / d, ny / d, 0.0])


def rotation_between(p0: np.ndarray, p1: np.ndarray, sensitivity: float = 1.0) -> Quaternion:
    """Quatérnio unitário que leva a direção p0 até p1."""
    axis = np.cross(p0, p1)
    denom = np.linalg.norm(p0) * np.linalg.norm(p1)
    if denom < 1e-12 or np.linalg.norm(axis) < 1e-9:
        return Quaternion.identity()
    cos_theta = float(np.clip(np.dot(p0, p1) / denom, -1.0, 1.0))
    theta = math.acos(cos_theta)
    return Quaternion.from_axis_angle(axis, theta * sensitivity)


class Trackball:
    """Mantém a orientação acumulada do objeto como um quatérnio unitário."""

    def __init__(self, sensitivity: float = 1.5):
        self.sensitivity = sensitivity
        self.orientation = Quaternion.identity()
        self._last: np.ndarray | None = None
        self.last_drag = Quaternion.identity()

    @property
    def dragging(self) -> bool:
        return self._last is not None

    def begin(self, x: float, y: float, width: int, height: int) -> None:
        self._last = project_to_sphere(*screen_to_ndc(x, y, width, height))

    def drag(self, x: float, y: float, width: int, height: int) -> Quaternion:
        if self._last is None:
            return self.orientation
        current = project_to_sphere(*screen_to_ndc(x, y, width, height))
        q_drag = rotation_between(self._last, current, self.sensitivity)
        # A rotação nova é aplicada DEPOIS da acumulada (eixos da câmera).
        # Renormalizar evita que erros de ponto flutuante "estiquem" o objeto.
        self.orientation = (q_drag * self.orientation).normalized()
        self.last_drag = q_drag
        self._last = current
        return self.orientation

    def end(self) -> None:
        self._last = None

    def reset(self) -> None:
        self.orientation = Quaternion.identity()
        self._last = None

    def matrix(self) -> np.ndarray:
        return self.orientation.to_matrix()
