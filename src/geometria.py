"""
PARTE 1 e PARTE 3 - Sólidos definidos por vértices e faces + vetores normais.
(Aulas 6_0, 6_1 e 2_0)

Cada sólido tem:
  - uma lista de VÉRTICES (x, y, z), centrados na origem;
  - uma lista de FACES, com os índices dos vértices de cada face.

Todas as faces estão em sentido ANTI-HORÁRIO quando vistas de fora.
Assim, pela regra da mão direita, a normal N = (V1 - V0) x (V2 - V0)
aponta para FORA do sólido.
"""
import numpy as np
from OpenGL.GL import (GL_LINES, GL_QUADS, GL_TRIANGLES, glBegin, glColor3f,
                       glEnd, glLineWidth, glNormal3fv, glVertex3fv)

# --------------------------------------------------------------------------
# Pirâmide de base quadrada (4 triângulos + 1 quadrado na base)
# --------------------------------------------------------------------------
PIRAMIDE_VERTICES = [
    (-0.8, -0.4, 0.8),   # 0 frente-esquerda
    (0.8, -0.4, 0.8),    # 1 frente-direita
    (0.8, -0.4, -0.8),   # 2 trás-direita
    (-0.8, -0.4, -0.8),  # 3 trás-esquerda
    (0.0, 1.2, 0.0),     # 4 topo
]
PIRAMIDE_FACES = [
    (0, 1, 4),       # frente
    (1, 2, 4),       # direita
    (2, 3, 4),       # trás
    (3, 0, 4),       # esquerda
    (0, 3, 2, 1),    # base (quadrado -> GL_QUADS)
]

# --------------------------------------------------------------------------
# Octaedro (8 triângulos)
# --------------------------------------------------------------------------
OCTAEDRO_VERTICES = [
    (1.2, 0, 0), (-1.2, 0, 0),   # 0 +x   1 -x
    (0, 1.2, 0), (0, -1.2, 0),   # 2 +y   3 -y
    (0, 0, 1.2), (0, 0, -1.2),   # 4 +z   5 -z
]
OCTAEDRO_FACES = [
    (4, 0, 2), (0, 5, 2), (5, 1, 2), (1, 4, 2),   # metade de cima
    (0, 4, 3), (5, 0, 3), (1, 5, 3), (4, 1, 3),   # metade de baixo
]

# --------------------------------------------------------------------------
# Tetraedro regular (4 triângulos)
# --------------------------------------------------------------------------
TETRAEDRO_VERTICES = [
    (0.75, 0.75, 0.75),
    (-0.75, -0.75, 0.75),
    (-0.75, 0.75, -0.75),
    (0.75, -0.75, -0.75),
]
TETRAEDRO_FACES = [
    (0, 2, 1),
    (0, 1, 3),
    (0, 3, 2),
    (1, 2, 3),
]

SOLIDOS = [
    ("Pirâmide", PIRAMIDE_VERTICES, PIRAMIDE_FACES),
    ("Octaedro", OCTAEDRO_VERTICES, OCTAEDRO_FACES),
    ("Tetraedro", TETRAEDRO_VERTICES, TETRAEDRO_FACES),
]


# --------------------------------------------------------------------------
# Normais (Parte 3)
# --------------------------------------------------------------------------
def calcular_normal(v0, v1, v2):
    """N = (V1 - V0) x (V2 - V0), normalizado para ter tamanho 1."""
    v0, v1, v2 = np.array(v0), np.array(v1), np.array(v2)
    normal = np.cross(v1 - v0, v2 - v0)
    return normal / np.linalg.norm(normal)


def calcular_normais(vertices, faces):
    """Uma normal para cada face (usa os 3 primeiros vértices da face)."""
    normais = []
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        normais.append(calcular_normal(v0, v1, v2))
    return normais


def centro_da_face(vertices, face):
    pontos = [np.array(vertices[i]) for i in face]
    return sum(pontos) / len(pontos)


# --------------------------------------------------------------------------
# Desenho simples: uma cor por face (sombreamento FLAT)
# --------------------------------------------------------------------------
def desenhar_solido(vertices, faces, normais):
    """Triângulos com GL_TRIANGLES e quadrados com GL_QUADS.
    Antes dos vértices de cada face informamos sua normal (glNormal3fv),
    que o OpenGL usa no cálculo da iluminação."""
    for i, face in enumerate(faces):
        if len(face) == 3:
            glBegin(GL_TRIANGLES)
        else:
            glBegin(GL_QUADS)
        glNormal3fv(normais[i])
        for indice in face:
            glVertex3fv(vertices[indice])
        glEnd()


# --------------------------------------------------------------------------
# Desenho detalhado: mesma face, dividida em triângulos menores
# --------------------------------------------------------------------------
def dividir_triangulo(a, b, c, niveis):
    """Divide o triângulo ABC em 4 triângulos menores usando os pontos médios
    dos lados, e repete isso 'niveis' vezes.

    Por quê? O OpenGL calcula a luz só nos VÉRTICES. Com poucos vértices, o
    brilho especular não aparece no meio da face. Com mais vértices (mas a
    MESMA normal), a luz é calculada em mais pontos e o brilho aparece."""
    if niveis == 0:
        return [(a, b, c)]
    ab = (a + b) / 2
    bc = (b + c) / 2
    ca = (c + a) / 2
    triangulos = []
    triangulos += dividir_triangulo(a, ab, ca, niveis - 1)
    triangulos += dividir_triangulo(ab, b, bc, niveis - 1)
    triangulos += dividir_triangulo(ca, bc, c, niveis - 1)
    triangulos += dividir_triangulo(ab, bc, ca, niveis - 1)
    return triangulos


def preparar_faces_detalhadas(vertices, faces, niveis=4):
    """Para cada face, gera a lista de triângulos menores (feito uma vez só,
    no início do programa). Um quadrado vira 2 triângulos antes de dividir."""
    faces_detalhadas = []
    for face in faces:
        pontos = [np.array(vertices[i], dtype=np.float32) for i in face]
        triangulos = dividir_triangulo(pontos[0], pontos[1], pontos[2], niveis)
        if len(face) == 4:
            triangulos += dividir_triangulo(pontos[0], pontos[2], pontos[3], niveis)
        faces_detalhadas.append(triangulos)
    return faces_detalhadas


def desenhar_solido_detalhado(faces_detalhadas, normais):
    glBegin(GL_TRIANGLES)
    for i, triangulos in enumerate(faces_detalhadas):
        glNormal3fv(normais[i])
        for a, b, c in triangulos:
            glVertex3fv(a)
            glVertex3fv(b)
            glVertex3fv(c)
    glEnd()


# --------------------------------------------------------------------------
# Auxiliares visuais (desenhados SEM iluminação)
# --------------------------------------------------------------------------
def desenhar_normais(vertices, faces, normais, tamanho=0.5):
    """Linha amarela saindo do centro de cada face na direção da normal."""
    glLineWidth(2.5)
    glColor3f(1.0, 0.85, 0.1)
    glBegin(GL_LINES)
    for i, face in enumerate(faces):
        centro = centro_da_face(vertices, face)
        glVertex3fv(centro)
        glVertex3fv(centro + normais[i] * tamanho)
    glEnd()


def desenhar_eixos(tamanho=1.8):
    """Eixos do mundo: X vermelho, Y verde, Z azul."""
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glColor3f(1, 0.25, 0.25)
    glVertex3fv((0, 0, 0))
    glVertex3fv((tamanho, 0, 0))
    glColor3f(0.25, 0.9, 0.35)
    glVertex3fv((0, 0, 0))
    glVertex3fv((0, tamanho, 0))
    glColor3f(0.3, 0.5, 1)
    glVertex3fv((0, 0, 0))
    glVertex3fv((0, 0, tamanho))
    glEnd()
