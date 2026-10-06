"""
PARTE 4 - Quatérnios e Trackball Virtual.
(Aula 7_0)

Um quatérnio é guardado como uma tupla  q = (w, x, y, z).
Para uma rotação de ângulo θ em torno de um eixo unitário (ex, ey, ez):

    q = ( cos(θ/2),  ex·sen(θ/2),  ey·sen(θ/2),  ez·sen(θ/2) )

Por que não usar só glRotatef em X, Y e Z (ângulos de Euler)?
Porque, dependendo da ordem, dois eixos podem se alinhar e perde-se um
grau de liberdade: é a TRAVA DE CARDÃ (Gimbal Lock). O quatérnio guarda a
rotação como UM eixo e UM ângulo, então esse problema não acontece.
"""
import math

import numpy as np

IDENTIDADE = (1.0, 0.0, 0.0, 0.0)  # "nenhuma rotação"


def quaternio_de_eixo_angulo(eixo, angulo_radianos):
    ex, ey, ez = eixo
    tamanho = math.sqrt(ex * ex + ey * ey + ez * ez)
    if tamanho == 0:
        return IDENTIDADE
    ex, ey, ez = ex / tamanho, ey / tamanho, ez / tamanho
    s = math.sin(angulo_radianos / 2)
    return (math.cos(angulo_radianos / 2), ex * s, ey * s, ez * s)


def multiplicar(q1, q2):
    """Produto de Hamilton: q1 * q2 aplica q2 primeiro e depois q1.
    (Assim como nas matrizes, a ordem importa.)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def normalizar(q):
    """Mantém |q| = 1. Depois de muitas multiplicações o erro de
    arredondamento se acumula; normalizar corrige isso."""
    w, x, y, z = q
    tamanho = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / tamanho, x / tamanho, y / tamanho, z / tamanho)


def quaternio_para_matriz(q):
    """Converte o quatérnio unitário em matriz de rotação 4x4
    (que entra no lugar de R em M = T @ R @ S)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


# --------------------------------------------------------------------------
# Trackball virtual
# --------------------------------------------------------------------------
def mapear_para_esfera(x, y, largura, altura):
    """Converte a posição do mouse (pixels) em um ponto (x, y, z) sobre um
    hemisfério de raio 1 virado para quem olha a tela.

    1) Leva (x, y) para o intervalo [-1, 1], com y para cima.
    2) Se o ponto cai dentro do círculo:  z = sqrt(1 - x² - y²)
       Se cai fora: fica na borda (z = 0)."""
    raio = min(largura, altura) / 2
    px = (x - largura / 2) / raio
    py = (altura / 2 - y) / raio   # na tela o y cresce para baixo

    d2 = px * px + py * py
    if d2 <= 1:
        pz = math.sqrt(1 - d2)
    else:
        d = math.sqrt(d2)
        px, py, pz = px / d, py / d, 0.0
    return np.array([px, py, pz])


def rotacao_do_arraste(p_inicio, p_fim, sensibilidade=1.5):
    """Quatérnio que gira p_inicio até p_fim:
       eixo   = p_inicio x p_fim         (produto vetorial)
       ângulo = acos(p_inicio · p_fim)   (produto escalar)"""
    eixo = np.cross(p_inicio, p_fim)
    if np.linalg.norm(eixo) < 1e-9:  # mouse não se moveu
        return IDENTIDADE
    cosseno = np.dot(p_inicio, p_fim) / (np.linalg.norm(p_inicio) * np.linalg.norm(p_fim))
    angulo = math.acos(max(-1.0, min(1.0, cosseno)))
    return quaternio_de_eixo_angulo(eixo, angulo * sensibilidade)
