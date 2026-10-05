"""
Painel de informações sobreposto à cena.

O texto é desenhado com OpenCV (cv2.putText) numa imagem RGBA, enviada como
textura e exibida num retângulo em projeção ortográfica — assim o projeto não
depende de GLUT para escrever na tela.
"""
from __future__ import annotations

import unicodedata

import cv2
import numpy as np
from OpenGL.GL import (
    GL_BLEND, GL_CLAMP_TO_EDGE, GL_DEPTH_TEST, GL_LIGHTING, GL_LINEAR,
    GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA, GL_PROJECTION, GL_QUADS, GL_RGBA,
    GL_SRC_ALPHA, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE,
    glBegin, glBindTexture, glBlendFunc, glColor4f, glDisable, glEnable,
    glEnd, glGenTextures, glLoadIdentity, glMatrixMode, glOrtho,
    glPixelStorei, glPopMatrix, glPushMatrix, glTexCoord2f, glTexImage2D,
    glTexParameteri, glVertex2f,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _ascii(text: str) -> str:
    """As fontes Hershey do OpenCV não têm acentos: 'Iluminação' → 'Iluminacao'."""
    text = text.replace("°", " graus").replace("·", ".").replace("→", "->")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


class Hud:
    def __init__(self, scale: float = 0.5):
        self.scale = scale
        self.texture = None
        self._key = None
        self._size = (0, 0)

    def _render_text(self, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
        line_h = int(30 * self.scale / 0.5 * 0.75)
        pad = 10
        widths = [cv2.getTextSize(_ascii(t), FONT, self.scale, 1)[0][0] for t, _ in lines] or [0]
        w = max(widths) + 2 * pad
        h = line_h * len(lines) + 2 * pad
        img = np.zeros((h, w, 4), dtype=np.uint8)
        img[..., :3] = (14, 16, 22)
        img[..., 3] = 190  # fundo semitransparente
        for i, (text, rgb) in enumerate(lines):
            y = pad + line_h * (i + 1) - 6
            cv2.putText(img, _ascii(text), (pad, y), FONT, self.scale,
                        (*rgb, 255), 1, cv2.LINE_AA)
        return img

    def draw(self, lines, fb_w: int, fb_h: int, corner: str = "top-left") -> None:
        if not lines:
            return
        key = tuple(lines)
        if key != self._key:
            img = self._render_text(lines)
            if self.texture is None:
                self.texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            for p, v in ((GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                         (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)):
                glTexParameteri(GL_TEXTURE_2D, p, v)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.shape[1], img.shape[0], 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, img)
            self._size = (img.shape[1], img.shape[0])
            self._key = key

        w, h = self._size
        x0 = 12 if corner.endswith("left") else fb_w - w - 12
        y0 = fb_h - h - 12 if corner.startswith("top") else 12

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, fb_w, 0, fb_h, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, self.texture)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        # a imagem do OpenCV tem a linha 0 no topo → t = 0 no topo do retângulo
        glTexCoord2f(0, 1); glVertex2f(x0, y0)
        glTexCoord2f(1, 1); glVertex2f(x0 + w, y0)
        glTexCoord2f(1, 0); glVertex2f(x0 + w, y0 + h)
        glTexCoord2f(0, 0); glVertex2f(x0, y0 + h)
        glEnd()
        glDisable(GL_TEXTURE_2D); glDisable(GL_BLEND); glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW); glPopMatrix()
