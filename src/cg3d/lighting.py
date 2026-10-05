"""
PARTE 3 — Iluminação e sombreamento com o modelo de Phong do OpenGL fixo.

    I = I_ambiente·k_a + I_difusa·k_d·max(N·L, 0) + I_especular·k_s·max(R·V, 0)^n

* Ambiente: luz "espalhada" que ilumina todas as faces igualmente.
* Difusa (Lambert): depende do ângulo entre a NORMAL da face e a direção da
  luz — é ela que faz cada face ter um tom diferente.
* Especular: o brilho que depende também da posição do observador.

Por isso as normais (Parte 3) são essenciais: sem elas a difusa e a
especular não têm como ser calculadas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from OpenGL.GL import (
    GL_AMBIENT, GL_COLOR_MATERIAL, GL_DIFFUSE, GL_FRONT_AND_BACK, GL_LIGHT0,
    GL_LIGHT_MODEL_AMBIENT, GL_LIGHTING, GL_NORMALIZE, GL_POSITION,
    GL_SHININESS, GL_SPECULAR, glDisable, glEnable, glLightfv,
    glLightModelfv, glMaterialf, glMaterialfv,
)


@dataclass
class Light:
    position: list[float] = field(default_factory=lambda: [3.0, 4.0, 5.0, 1.0])  # w=1: pontual
    ambient: list[float] = field(default_factory=lambda: [0.30, 0.30, 0.34, 1.0])
    diffuse: list[float] = field(default_factory=lambda: [0.95, 0.95, 0.90, 1.0])
    specular: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    enabled: bool = True


@dataclass
class Material:
    color: tuple[float, float, float] = (0.15, 0.45, 0.95)
    ambient_factor: float = 0.6
    specular: tuple[float, float, float] = (0.7, 0.7, 0.7)
    shininess: float = 48.0


def setup_lighting(light: Light) -> None:
    """Habilita o sistema de iluminação (GL_LIGHTING + GL_LIGHT0) e define as
    componentes ambiente, difusa e especular da luz."""
    if not light.enabled:
        glDisable(GL_LIGHTING)
        return
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    # GL_NORMALIZE: renormaliza as normais depois da Matriz de Modelo, pois a
    # escala S altera o comprimento delas e quebraria |N| = 1.
    glEnable(GL_NORMALIZE)
    glDisable(GL_COLOR_MATERIAL)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.05, 0.05, 0.05, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, light.ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light.diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light.specular)


def place_light(light: Light) -> None:
    """Deve ser chamada DEPOIS de carregar a Matriz de Visão: o OpenGL
    transforma a posição da luz pela MODELVIEW corrente, então a luz fica
    fixa no Espaço do Mundo (não gira junto com o objeto)."""
    glLightfv(GL_LIGHT0, GL_POSITION, light.position)


def apply_material(material: Material) -> None:
    r, g, b = material.color
    a = material.ambient_factor
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [r * a, g * a, b * a, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [r, g, b, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [*material.specular, 1.0])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material.shininess)
