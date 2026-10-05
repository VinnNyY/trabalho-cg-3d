"""
PARTE 3 — Iluminação e sombreamento (modelo de Phong / Blinn-Phong do OpenGL).

Para cada vértice o pipeline fixo calcula (com tudo em [0, 1]):

    I = A_global·k_a
      + A_luz·k_a                                   (componente AMBIENTE)
      + D_luz·k_d · max(N·L, 0)                     (componente DIFUSA, Lambert)
      + S_luz·k_s · max(N·H, 0)^n   se N·L > 0      (componente ESPECULAR)

    N = normal unitária da face        L = direção vértice → luz
    V = direção vértice → observador   H = normalize(L + V)  (vetor "meio-caminho")
    n = brilho (GL_SHININESS)          k_a, k_d, k_s = coeficientes do material

* AMBIENTE: luz indireta; ilumina todas as faces igualmente (sem ela, faces de
  costas para a luz ficariam pretas).
* DIFUSA: depende só do ângulo entre N e L — é o que dá um tom diferente a
  cada face e mostra o volume do objeto.
* ESPECULAR: o reflexo brilhante; depende também de onde está o observador.

``phong_shade()`` reproduz essa equação em NumPy. Ela é usada no painel de
informações (N·L de cada face em tempo real) e nos testes, que comparam o
resultado com os pixels que o próprio OpenGL desenhou.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from OpenGL.GL import (
    GL_AMBIENT, GL_COLOR_MATERIAL, GL_CONSTANT_ATTENUATION, GL_DIFFUSE,
    GL_FRONT_AND_BACK, GL_LIGHT0, GL_LIGHT_MODEL_AMBIENT,
    GL_LIGHT_MODEL_LOCAL_VIEWER, GL_LIGHT_MODEL_TWO_SIDE, GL_LIGHTING,
    GL_LINEAR_ATTENUATION, GL_NORMALIZE, GL_POSITION,
    GL_QUADRATIC_ATTENUATION, GL_SHININESS, GL_SPECULAR, GL_TRUE, GL_FALSE,
    glDisable, glEnable, glLightf, glLightfv, glLightModelfv, glLightModeli,
    glMaterialf, glMaterialfv,
)

GLOBAL_AMBIENT = (0.04, 0.04, 0.05)

# Cores de luz para demonstrar a mistura cor-da-luz × cor-do-material.
LIGHT_COLORS = {
    "Branca": (1.00, 1.00, 1.00),
    "Quente": (1.00, 0.78, 0.50),
    "Fria": (0.55, 0.75, 1.00),
    "Verde": (0.50, 1.00, 0.55),
}


@dataclass
class Light:
    """Fonte de luz GL_LIGHT0.

    ``position`` está no Espaço do Mundo. w = 1 → luz PONTUAL (os raios saem
    de um ponto e L muda ao longo da face); w = 0 → luz DIRECIONAL (raios
    paralelos, como o Sol; L é igual em toda a cena)."""
    position: list[float] = field(default_factory=lambda: [2.0, 1.3, 0.8, 1.0])
    color_name: str = "Branca"
    ambient_intensity: float = 0.30
    diffuse_intensity: float = 1.00
    specular_intensity: float = 1.00
    use_ambient: bool = True
    use_diffuse: bool = True
    use_specular: bool = True
    enabled: bool = True

    @property
    def color(self) -> tuple[float, float, float]:
        return LIGHT_COLORS[self.color_name]

    @property
    def directional(self) -> bool:
        return self.position[3] == 0.0

    def _component(self, on: bool, intensity: float) -> list[float]:
        if not on:
            return [0.0, 0.0, 0.0, 1.0]
        return [c * intensity for c in self.color] + [1.0]

    @property
    def ambient(self) -> list[float]:
        return self._component(self.use_ambient, self.ambient_intensity)

    @property
    def diffuse(self) -> list[float]:
        return self._component(self.use_diffuse, self.diffuse_intensity)

    @property
    def specular(self) -> list[float]:
        return self._component(self.use_specular, self.specular_intensity)

    def next_color(self) -> None:
        names = list(LIGHT_COLORS)
        self.color_name = names[(names.index(self.color_name) + 1) % len(names)]

    def toggle_type(self) -> None:
        self.position[3] = 0.0 if self.position[3] == 1.0 else 1.0

    def components_label(self) -> str:
        on = lambda b: "ON " if b else "off"  # noqa: E731
        return (f"Ambiente {on(self.use_ambient)} | Difusa {on(self.use_diffuse)} | "
                f"Especular {on(self.use_specular)}")


@dataclass
class Material:
    color: tuple[float, float, float] = (0.15, 0.45, 0.95)  # k_d
    ambient_factor: float = 0.6                              # k_a = fator · k_d
    specular: tuple[float, float, float] = (0.65, 0.65, 0.65)  # k_s
    shininess: float = 60.0                                  # n (0..128)

    @property
    def ambient(self) -> tuple[float, float, float]:
        return tuple(c * self.ambient_factor for c in self.color)


# --------------------------------------------------------------------------- #
# Configuração do OpenGL
# --------------------------------------------------------------------------- #
def setup_lighting(light: Light) -> None:
    """Habilita GL_LIGHTING e GL_LIGHT0 e define as componentes ambiente,
    difusa e especular da luz."""
    if not light.enabled:
        glDisable(GL_LIGHTING)
        return
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    # GL_NORMALIZE: a escala S altera o comprimento das normais; o OpenGL as
    # renormaliza depois da Matriz de Modelo para manter |N| = 1.
    glEnable(GL_NORMALIZE)
    glDisable(GL_COLOR_MATERIAL)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [*GLOBAL_AMBIENT, 1.0])
    # Observador local: V é calculado para cada vértice (especular realista).
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light.ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light.diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light.specular)
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.0)
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.0)


def place_light(light: Light) -> None:
    """Chamada DEPOIS de carregar a Matriz de Visão: o OpenGL multiplica a
    posição da luz pela MODELVIEW corrente, então a luz fica fixa no Mundo
    (não gira junto com o objeto)."""
    glLightfv(GL_LIGHT0, GL_POSITION, light.position)


def apply_material(material: Material) -> None:
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [*material.ambient, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [*material.color, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [*material.specular, 1.0])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, float(np.clip(material.shininess, 0, 128)))


# --------------------------------------------------------------------------- #
# A mesma equação, em NumPy
# --------------------------------------------------------------------------- #
@dataclass
class ShadeResult:
    color: np.ndarray        # cor final RGB (limitada a [0, 1])
    ambient: np.ndarray      # parcela ambiente
    diffuse: np.ndarray      # parcela difusa
    specular: np.ndarray     # parcela especular
    n_dot_l: float
    n_dot_h: float


def light_direction(light: Light, point) -> np.ndarray:
    """L unitário, do ponto para a luz (pontual) ou a própria direção
    da luz (direcional, w = 0)."""
    p = np.asarray(light.position[:3], dtype=np.float64)
    if light.directional:
        return p / np.linalg.norm(p)
    d = p - np.asarray(point, dtype=np.float64)
    return d / np.linalg.norm(d)


def phong_shade(normal, point, eye, light: Light, material: Material) -> ShadeResult:
    """Equação de iluminação do OpenGL (Blinn-Phong, observador local)
    avaliada num ponto — tudo no Espaço do Mundo."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    l_vec = light_direction(light, point)
    v_vec = np.asarray(eye, dtype=np.float64) - np.asarray(point, dtype=np.float64)
    v_vec /= np.linalg.norm(v_vec)
    h_vec = l_vec + v_vec
    h_len = np.linalg.norm(h_vec)
    h_vec = h_vec / h_len if h_len > 1e-12 else np.zeros(3)  # L oposto a V: sem reflexo

    n_dot_l = float(np.dot(n, l_vec))
    n_dot_h = float(np.dot(n, h_vec))

    ka, kd, ks = (np.array(c, dtype=np.float64) for c in (material.ambient, material.color, material.specular))
    la, ld, ls = (np.array(c[:3]) for c in (light.ambient, light.diffuse, light.specular))

    ambient = np.array(GLOBAL_AMBIENT) * ka + la * ka
    diffuse = ld * kd * max(n_dot_l, 0.0)
    if n_dot_l > 0.0 and n_dot_h > 0.0:
        specular = ls * ks * (n_dot_h ** material.shininess)
    else:
        specular = np.zeros(3)
    color = np.clip(ambient + diffuse + specular, 0.0, 1.0)
    return ShadeResult(color, ambient, diffuse, specular, n_dot_l, n_dot_h)
