"""
PARTE 4 — Quatérnios unitários para representar rotações.

Um quatérnio  q = (w, x, y, z) = w + x·i + y·j + z·k
com |q| = 1 representa uma rotação de ângulo θ em torno do eixo unitário û:

    q = ( cos(θ/2),  sin(θ/2)·û )

Vantagens sobre ângulos de Euler:
* Não há trava de cardã (Gimbal Lock): a rotação é descrita por UM eixo e UM
  ângulo, não por três rotações encadeadas que podem alinhar dois eixos.
* Compor rotações = multiplicar quatérnios (16 multiplicações), e a
  renormalização elimina o acúmulo de erro numérico.

Composição
----------
    q_total = q2 * q1     →  aplica q1 primeiro e depois q2
(mesma convenção das matrizes: o que está à direita age primeiro).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class Quaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # ------------------------------------------------------------------ #
    # Construtores
    # ------------------------------------------------------------------ #
    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_axis_angle(axis, angle_rad: float) -> "Quaternion":
        """q = (cos(θ/2), sin(θ/2)·û). O eixo é normalizado aqui; um eixo nulo
        (θ indefinido) resulta na identidade."""
        a = np.asarray(axis, dtype=np.float64)
        n = np.linalg.norm(a)
        if n < _EPS:
            return Quaternion.identity()
        a = a / n
        half = angle_rad / 2.0
        s = math.sin(half)
        return Quaternion(math.cos(half), a[0] * s, a[1] * s, a[2] * s)

    # ------------------------------------------------------------------ #
    # Álgebra
    # ------------------------------------------------------------------ #
    def __mul__(self, o: "Quaternion") -> "Quaternion":
        """Produto de Hamilton (não comutativo):
        (w1, v1)(w2, v2) = (w1·w2 − v1·v2,  w1·v2 + w2·v1 + v1 × v2)."""
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = o.w, o.x, o.y, o.z
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def norm(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Quaternion":
        n = self.norm()
        if n < _EPS:
            return Quaternion.identity()
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def conjugate(self) -> "Quaternion":
        """Para quatérnio unitário, o conjugado é a inversa (rotação oposta)."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def rotate(self, v) -> np.ndarray:
        """Rotaciona o vetor v:  v' = q · (0, v) · q*."""
        p = Quaternion(0.0, float(v[0]), float(v[1]), float(v[2]))
        r = self * p * self.conjugate()
        return np.array([r.x, r.y, r.z])

    def to_axis_angle(self):
        q = self.normalized()
        w = max(-1.0, min(1.0, q.w))
        angle = 2.0 * math.acos(w)
        s = math.sqrt(max(0.0, 1.0 - w * w))
        if s < 1e-9:
            return np.array([1.0, 0.0, 0.0]), 0.0
        return np.array([q.x / s, q.y / s, q.z / s]), angle

    # ------------------------------------------------------------------ #
    # Conversão para o pipeline
    # ------------------------------------------------------------------ #
    def to_matrix(self) -> np.ndarray:
        """Matriz de rotação 4x4 (homogênea) equivalente ao quatérnio unitário:

            | 1-2(y²+z²)   2(xy-wz)    2(xz+wy)   0 |
            | 2(xy+wz)     1-2(x²+z²)  2(yz-wx)   0 |
            | 2(xz-wy)     2(yz+wx)    1-2(x²+y²) 0 |
            | 0            0           0          1 |
        """
        q = self.normalized()
        w, x, y, z = q.w, q.x, q.y, q.z
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0.0],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0.0],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def as_tuple(self):
        return (self.w, self.x, self.y, self.z)

    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"
