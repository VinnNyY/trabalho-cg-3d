"""
PARTE 3 — Modelos de cor (RGB ↔ HSV) e análise de cor com OpenCV.

Convenções
----------
* No OpenGL as cores são RGB em [0, 1].
* No OpenCV (8 bits): imagens em ordem **BGR**, e o HSV usa
  H ∈ [0, 179] (graus / 2, para caber em 1 byte), S ∈ [0, 255], V ∈ [0, 255].

O módulo traz uma implementação MANUAL de RGB→HSV (as fórmulas da aula) e a
compara com ``cv2.cvtColor`` — ver ``tests/test_color.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# RGB ↔ HSV (implementação própria)
# --------------------------------------------------------------------------- #
def rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    """RGB em [0,1] → (H em graus [0,360), S em [0,1], V em [0,1]).

        V = max(R,G,B)
        C = V − min(R,G,B)                (croma)
        S = C / V            (S = 0 se V = 0)
        H = 60° · { (G−B)/C mod 6   se V = R
                    (B−R)/C + 2     se V = G
                    (R−G)/C + 4     se V = B }
    """
    cmax, cmin = max(r, g, b), min(r, g, b)
    c = cmax - cmin
    v = cmax
    s = 0.0 if cmax == 0 else c / cmax
    if c == 0:
        h = 0.0
    elif cmax == r:
        h = 60.0 * (((g - b) / c) % 6)
    elif cmax == g:
        h = 60.0 * ((b - r) / c + 2)
    else:
        h = 60.0 * ((r - g) / c + 4)
    return h % 360.0, s, v


def hsv_to_rgb(h_deg: float, s: float, v: float) -> tuple[float, float, float]:
    """(H em graus, S, V) → RGB em [0,1]  (inversa de rgb_to_hsv)."""
    c = v * s
    hp = (h_deg % 360.0) / 60.0
    x = c * (1 - abs(hp % 2 - 1))
    m = v - c
    if hp < 1:
        r, g, b = c, x, 0
    elif hp < 2:
        r, g, b = x, c, 0
    elif hp < 3:
        r, g, b = 0, c, x
    elif hp < 4:
        r, g, b = 0, x, c
    elif hp < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return r + m, g + m, b + m


def rgb_to_hsv_opencv(r: float, g: float, b: float) -> tuple[int, int, int]:
    """Mesma conversão feita pelo OpenCV, para comparação (escala OpenCV)."""
    pixel = np.uint8([[[round(b * 255), round(g * 255), round(r * 255)]]])
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    return int(h), int(s), int(v)


def hsv_degrees_to_opencv(h_deg: float, s: float, v: float) -> tuple[float, float, float]:
    return h_deg / 2.0, s * 255.0, v * 255.0


# --------------------------------------------------------------------------- #
# Análise de cor com OpenCV
# --------------------------------------------------------------------------- #
def make_palette_image(width: int = 360, height: int = 200) -> np.ndarray:
    """Imagem sintética (BGR) para a análise: matiz varia no eixo X
    (0 → 179) e saturação no eixo Y (255 no topo → 0 embaixo), com V = 255.
    Funciona como uma "roda de cores" aberta, ideal para testar limites HSV."""
    hue = np.linspace(0, 179, width, dtype=np.float32)
    sat = np.linspace(255, 0, height, dtype=np.float32)
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 0] = hue[np.newaxis, :].astype(np.uint8)
    hsv[..., 1] = sat[:, np.newaxis].astype(np.uint8)
    hsv[..., 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@dataclass
class HSVRange:
    """Limites HSV na escala do OpenCV. Se h_min > h_max a faixa "dá a volta"
    pelo vermelho (ex.: 170 → 10), tratada com duas máscaras."""
    h_min: int = 100
    h_max: int = 130
    s_min: int = 120
    s_max: int = 255
    v_min: int = 70
    v_max: int = 255

    def shift_hue(self, delta: int) -> None:
        self.h_min = (self.h_min + delta) % 180
        self.h_max = (self.h_max + delta) % 180

    def widen(self, delta: int) -> None:
        width = (self.h_max - self.h_min) % 180
        new_width = int(np.clip(width + 2 * delta, 4, 170))
        center = (self.h_min + width / 2.0) % 180
        self.h_min = int(round(center - new_width / 2.0)) % 180
        self.h_max = int(round(center + new_width / 2.0)) % 180

    def shift_saturation(self, delta: int) -> None:
        self.s_min = int(np.clip(self.s_min + delta, 0, 250))

    def __str__(self) -> str:
        return f"H[{self.h_min}-{self.h_max}] S[{self.s_min}-{self.s_max}] V[{self.v_min}-{self.v_max}]"


def hsv_mask(image_bgr: np.ndarray, rng: HSVRange) -> np.ndarray:
    """Máscara binária (0/255) dos pixels cuja cor está dentro dos limites."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if rng.h_min <= rng.h_max:
        lower = np.array([rng.h_min, rng.s_min, rng.v_min], dtype=np.uint8)
        upper = np.array([rng.h_max, rng.s_max, rng.v_max], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)
    m1 = cv2.inRange(
        hsv,
        np.array([rng.h_min, rng.s_min, rng.v_min], dtype=np.uint8),
        np.array([179, rng.s_max, rng.v_max], dtype=np.uint8),
    )
    m2 = cv2.inRange(
        hsv,
        np.array([0, rng.s_min, rng.v_min], dtype=np.uint8),
        np.array([rng.h_max, rng.s_max, rng.v_max], dtype=np.uint8),
    )
    return cv2.bitwise_or(m1, m2)


def mask_mean_color(image_bgr: np.ndarray, mask: np.ndarray):
    """Cor média (RGB em [0,1]) dos pixels selecionados pela máscara.
    Retorna None se a máscara estiver vazia.

    Média feita no espaço circular do matiz seria mais exata para faixas que
    cruzam o vermelho, mas para o material da cena a média RGB é o que
    representa visualmente a "cor dominante" segmentada."""
    if cv2.countNonZero(mask) == 0:
        return None
    b, g, r, _ = cv2.mean(image_bgr, mask=mask)
    return (r / 255.0, g / 255.0, b / 255.0)


def analysis_preview(image_bgr: np.ndarray, mask: np.ndarray, color_rgb) -> np.ndarray:
    """Painel lado a lado: imagem | máscara | resultado segmentado | amostra
    da cor aplicada no material — útil para a demonstração em sala."""
    h, w = image_bgr.shape[:2]
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    swatch = np.zeros_like(image_bgr)
    if color_rgb is not None:
        swatch[:] = [int(c * 255) for c in reversed(color_rgb)]
    panel = np.hstack([image_bgr, mask_bgr, segmented, swatch])
    labels = ["Imagem", "Mascara HSV", "Segmentado", "Cor -> material"]
    for i, text in enumerate(labels):
        cv2.putText(panel, text, (i * w + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, text, (i * w + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return panel


__all__ = [
    "rgb_to_hsv", "hsv_to_rgb", "rgb_to_hsv_opencv", "hsv_degrees_to_opencv",
    "make_palette_image", "HSVRange", "hsv_mask", "mask_mean_color",
    "analysis_preview",
]
