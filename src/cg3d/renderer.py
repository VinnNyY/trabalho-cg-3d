"""
Desenho da malha e de elementos auxiliares (eixos, normais, marcador da luz)
com o pipeline fixo do OpenGL.
"""
from __future__ import annotations

import numpy as np
from OpenGL.GL import (
    GL_LIGHTING, GL_LINES, GL_POINTS, GL_QUADS, GL_TRIANGLES, glBegin,
    glColor3f, glDisable, glEnable, glEnd, glIsEnabled, glLineWidth,
    glNormal3fv, glPointSize, glVertex3fv,
)

from .geometry import Mesh


def draw_mesh(mesh: Mesh) -> None:
    """Renderização em malha poligonal:
    faces triangulares em GL_TRIANGLES e faces quadrilaterais em GL_QUADS.
    Cada face recebe sua normal via glNormal3fv() antes dos vértices
    (sombreamento facetado/flat)."""
    v = mesh.vertices.astype(np.float32)
    n = mesh.face_normals.astype(np.float32)

    glBegin(GL_TRIANGLES)
    for i in mesh.triangles:
        glNormal3fv(n[i])
        for idx in mesh.faces[i]:
            glVertex3fv(v[idx])
    glEnd()

    if mesh.quads:
        glBegin(GL_QUADS)
        for i in mesh.quads:
            glNormal3fv(n[i])
            for idx in mesh.faces[i]:
                glVertex3fv(v[idx])
        glEnd()


def _without_lighting(fn):
    def wrapper(*args, **kwargs):
        was_lit = glIsEnabled(GL_LIGHTING)
        glDisable(GL_LIGHTING)
        try:
            fn(*args, **kwargs)
        finally:
            if was_lit:
                glEnable(GL_LIGHTING)
    return wrapper


@_without_lighting
def draw_normals(mesh: Mesh, length: float = 0.5) -> None:
    """Segmentos amarelos saindo do centro de cada face na direção da normal."""
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glColor3f(1.0, 0.85, 0.1)
    for i in range(len(mesh.faces)):
        c = mesh.face_centroid(i)
        glVertex3fv(c.astype(np.float32))
        glVertex3fv((c + mesh.face_normals[i] * length).astype(np.float32))
    glEnd()


@_without_lighting
def draw_axes(length: float = 1.8) -> None:
    """Eixos do sistema de coordenadas: X vermelho, Y verde, Z azul."""
    glLineWidth(2.0)
    glBegin(GL_LINES)
    for axis, color in ((0, (0.95, 0.25, 0.25)), (1, (0.25, 0.9, 0.35)), (2, (0.3, 0.5, 1.0))):
        end = np.zeros(3, dtype=np.float32)
        end[axis] = length
        glColor3f(*color)
        glVertex3fv(np.zeros(3, dtype=np.float32))
        glVertex3fv(end)
    glEnd()


@_without_lighting
def draw_grid(size: float = 4.0, step: float = 0.5, y: float = -1.5) -> None:
    glLineWidth(1.0)
    glBegin(GL_LINES)
    glColor3f(0.28, 0.30, 0.36)
    k = -size
    while k <= size + 1e-6:
        glVertex3fv(np.array([k, y, -size], dtype=np.float32))
        glVertex3fv(np.array([k, y, size], dtype=np.float32))
        glVertex3fv(np.array([-size, y, k], dtype=np.float32))
        glVertex3fv(np.array([size, y, k], dtype=np.float32))
        k += step
    glEnd()


@_without_lighting
def draw_light_marker(position) -> None:
    glPointSize(12.0)
    glBegin(GL_POINTS)
    glColor3f(1.0, 0.95, 0.6)
    glVertex3fv(np.asarray(position[:3], dtype=np.float32))
    glEnd()
