"""
PARTE 3 - Modelos de cor (RGB e HSV) e análise com OpenCV.
(Aulas 5_0 e 5_1)

RGB: vermelho, verde e azul. No OpenGL cada canal vai de 0 a 1.
HSV: H = matiz (qual cor), S = saturação (quão pura), V = valor (brilho).

No OpenCV:
  - as imagens ficam em BGR (ordem invertida);
  - H vai de 0 a 179 (graus / 2), S e V vão de 0 a 255.
"""
import cv2
import numpy as np


def rgb_para_hsv(r, g, b):
    """Conversão manual (fórmulas da aula). Entrada RGB de 0 a 1.
    Saída: H em graus (0 a 360), S e V de 0 a 1."""
    maximo = max(r, g, b)
    minimo = min(r, g, b)
    delta = maximo - minimo

    v = maximo
    s = 0 if maximo == 0 else delta / maximo

    if delta == 0:
        h = 0
    elif maximo == r:
        h = 60 * (((g - b) / delta) % 6)
    elif maximo == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)
    return h, s, v


def hsv_para_rgb(h, s, v):
    """Caminho inverso: H em graus, S e V de 0 a 1 -> RGB de 0 a 1."""
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return r + m, g + m, b + m


def rgb_para_hsv_opencv(r, g, b):
    """A mesma conversão feita pelo OpenCV, para comparar com a nossa."""
    pixel = np.uint8([[[b * 255, g * 255, r * 255]]])  # OpenCV usa BGR
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    return int(h), int(s), int(v)


def criar_imagem_paleta(largura=360, altura=200):
    """Imagem com todas as cores: a matiz muda da esquerda para a direita e a
    saturação diminui de cima para baixo. Criamos em HSV e convertemos para BGR."""
    imagem_hsv = np.zeros((altura, largura, 3), dtype=np.uint8)
    for x in range(largura):
        imagem_hsv[:, x, 0] = int(x * 179 / (largura - 1))      # H
    for y in range(altura):
        imagem_hsv[y, :, 1] = int(255 - y * 255 / (altura - 1))  # S
    imagem_hsv[:, :, 2] = 255                                    # V
    return cv2.cvtColor(imagem_hsv, cv2.COLOR_HSV2BGR)


def criar_mascara(imagem_bgr, h_min, h_max, s_min=120):
    """Máscara: branco (255) onde a cor está dentro dos limites, preto (0) fora."""
    imagem_hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    limite_inferior = np.array([h_min, s_min, 70])
    limite_superior = np.array([h_max, 255, 255])
    return cv2.inRange(imagem_hsv, limite_inferior, limite_superior)


def cor_media_da_mascara(imagem_bgr, mascara):
    """Cor média (RGB de 0 a 1) dos pixels brancos da máscara.
    Essa cor vira o material do objeto 3D."""
    if cv2.countNonZero(mascara) == 0:
        return None
    b, g, r, _ = cv2.mean(imagem_bgr, mask=mascara)
    return r / 255, g / 255, b / 255


def montar_painel(imagem_bgr, mascara, cor):
    """Junta lado a lado: imagem | máscara | parte segmentada | cor escolhida."""
    mascara_colorida = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    segmentada = cv2.bitwise_and(imagem_bgr, imagem_bgr, mask=mascara)
    amostra = np.zeros_like(imagem_bgr)
    if cor is not None:
        r, g, b = cor
        amostra[:] = (int(b * 255), int(g * 255), int(r * 255))
    painel = np.hstack([imagem_bgr, mascara_colorida, segmentada, amostra])

    largura = imagem_bgr.shape[1]
    titulos = ["Imagem", "Mascara HSV", "Segmentado", "Cor do material"]
    for i, titulo in enumerate(titulos):
        cv2.putText(painel, titulo, (i * largura + 8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(painel, titulo, (i * largura + 8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return painel
