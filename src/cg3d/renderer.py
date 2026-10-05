"""
Desenho da malha e de elementos auxiliares (normais, fonte de luz, eixos)
com o pipeline fixo do OpenGL.
"""
from __future__ import annotations

import math

import numpy as np
from OpenGL.GL import (
    GL_FLAT, GL_FLOAT, GL_LIGHTING, GL_LINES, GL_NORMAL_ARRAY, GL_QUADS,
    GL_SMOOTH, GL_TRIANGLES, GL_VERTEX_ARRAY, glBegin, glColor3f,
    glDisable, glDisableClientState, glDrawArrays, glEnable,
    glEnableClientState, glEnd, glIsEnabled, glLineWidth, glNormal3fv,
    glNormalPointer, glShadeModel, glVertex3fv, glVertexPointer,
)

from .geometry import Mesh, subdivide_faces


def draw_mesh(mesh: Mesh) -> None:
    """Modo FLAT — malha original: faces triangulares em GL_TRIANGLES e
    quadrilaterais em GL_QUADS, uma normal por face via glNormal3fv().
    GL_FLAT faz a face inteira receber uma única cor."""
    glShadeModel(GL_FLAT)
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


_SUBDIV_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def draw_mesh_detailed(mesh: Mesh, level: int = 16) -> None:
    """Modo DETALHADO — mesmas faces e mesmas normais, mas subdivididas em
    triângulos menores (GL_TRIANGLES via vertex arrays) com GL_SMOOTH. A luz é
    avaliada em muitos pontos de cada face: aparece o degradê da luz pontual e
    o reflexo especular."""
    key = id(mesh) * 1000 + level
    if key not in _SUBDIV_CACHE:
        _SUBDIV_CACHE[key] = subdivide_faces(mesh, level)
    positions, normals = _SUBDIV_CACHE[key]

    glShadeModel(GL_SMOOTH)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, positions)
    glNormalPointer(GL_FLOAT, 0, normals)
    glDrawArrays(GL_TRIANGLES, 0, len(positions))
    glDisableClientState(GL_NORMAL_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)


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
def draw_normals(mesh: Mesh, length: float = 0.5, intensities=None) -> None:
    """Segmentos saindo do centro de cada face na direção da normal.
    Se ``intensities`` (N·L por face) for dado, a cor vai de cinza (de costas
    para a luz) a amarelo forte (de frente para a luz)."""
    glLineWidth(2.5)
    glBegin(GL_LINES)
    for i in range(len(mesh.faces)):
        if intensities is None:
            glColor3f(1.0, 0.85, 0.1)
        else:
            k = max(float(intensities[i]), 0.0)
            glColor3f(0.35 + 0.65 * k, 0.35 + 0.5 * k, 0.35 - 0.3 * k)
        c = mesh.face_centroid(i)
        glVertex3fv(c.astype(np.float32))
        glVertex3fv((c + mesh.face_normals[i] * length).astype(np.float32))
    glEnd()


@_without_lighting
def draw_axes(length: float = 1.8) -> None:
    """Eixos do mundo: X vermelho, Y verde, Z azul."""
    glLineWidth(2.0)
    glBegin(GL_LINES)
    for axis, color in ((0, (0.95, 0.25, 0.25)), (1, (0.25, 0.9, 0.35)), (2, (0.3, 0.5, 1.0))):
        end = np.zeros(3, dtype=np.float32)
        end[axis] = length
        glColor3f(*color)
        glVertex3fv(np.zeros(3, dtype=np.float32))
        glVertex3fv(end)
    glEnd()


def _sphere_triangles(radius: float, stacks: int = 12, slices: int = 18) -> np.ndarray:
    pts = []
    for i in range(stacks):
        t0, t1 = math.pi * i / stacks, math.pi * (i + 1) / stacks
        for j in range(slices):
            p0, p1 = 2 * math.pi * j / slices, 2 * math.pi * (j + 1) / slices
            q = lambda t, p: (radius * math.sin(t) * math.cos(p), radius * math.cos(t),  # noqa: E731
                              radius * math.sin(t) * math.sin(p))
            pts += [q(t0, p0), q(t1, p0), q(t1, p1), q(t0, p0), q(t1, p1), q(t0, p1)]
    return np.asarray(pts, dtype=np.float32)


_SPHERE = _sphere_triangles(0.12)


@_without_lighting
def draw_light(position, color, target=(0.0, 0.0, 0.0)) -> None:
    """Fonte de luz visível: uma pequena esfera emissiva na cor da luz e um
    raio tracejado até o objeto (luz pontual) — ou uma seta indicando a
    direção dos raios paralelos (luz direcional)."""
    pos = np.asarray(position[:3], dtype=np.float32)
    directional = position[3] == 0.0
    if directional:
        # luz direcional não tem posição: desenha a "seta do Sol" longe do objeto
        d = pos / np.linalg.norm(pos)
        pos = (d * 3.2).astype(np.float32)

    glColor3f(*color)
    glBegin(GL_TRIANGLES)
    for p in _SPHERE:
        glVertex3fv(p + pos)
    glEnd()

    glLineWidth(1.5)
    glBegin(GL_LINES)
    tgt = np.asarray(target, dtype=np.float32)
    seg = 14
    for k in range(0, seg, 2):  # tracejado
        a = pos + (tgt - pos) * (k / seg)
        b = pos + (tgt - pos) * ((k + 1) / seg)
        glColor3f(*(0.6 * np.asarray(color)))
        glVertex3fv(a)
        glVertex3fv(b)
    glEnd()
