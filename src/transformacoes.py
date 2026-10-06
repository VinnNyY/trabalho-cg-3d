"""
PARTE 2 - Transformações geométricas 3D em coordenadas homogêneas.
(Aulas 3_0 e 4_0)

Cada transformação é uma matriz 4x4. Um ponto (x, y, z) vira (x, y, z, 1)
e é transformado por:  ponto_novo = M @ ponto

Composição: M = T @ R @ S
A matriz mais à DIREITA é aplicada PRIMEIRO ao objeto:
    1º escala (S), 2º rotação (R), 3º translação (T).
Assim o objeto escala e gira em torno do próprio centro e só depois
é levado para a posição final.
"""
import math

import numpy as np
from OpenGL.GL import glMultMatrixf


def matriz_translacao(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def matriz_escala(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def matriz_rotacao_x(angulo_graus):
    a = math.radians(angulo_graus)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def matriz_rotacao_y(angulo_graus):
    a = math.radians(angulo_graus)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def matriz_rotacao_z(angulo_graus):
    a = math.radians(angulo_graus)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def matriz_modelo(T, R, S):
    """Matriz de Modelo: leva o objeto do Espaço do Objeto (local)
    para o Espaço do Mundo.  M = T @ R @ S"""
    return T @ R @ S


def transformar_ponto(M, ponto):
    """Aplica a matriz M em um ponto 3D (usado nos testes e nos cálculos)."""
    x, y, z = ponto
    resultado = M @ np.array([x, y, z, 1], dtype=np.float32)
    return resultado[:3]


def enviar_para_opengl(M):
    """Multiplica a matriz atual do OpenGL (que já tem a câmera do gluLookAt)
    pela nossa matriz de modelo.

    O NumPy guarda a matriz por LINHAS e o OpenGL lê por COLUNAS,
    por isso enviamos a TRANSPOSTA (M.T)."""
    glMultMatrixf(np.ascontiguousarray(M.T, dtype=np.float32))
