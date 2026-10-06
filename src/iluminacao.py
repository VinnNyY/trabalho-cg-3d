"""
PARTE 3 - Iluminação e sombreamento (modelo de Phong do OpenGL).

O OpenGL soma três componentes em cada vértice:

  AMBIENTE : luz "espalhada" pelo ambiente. Ilumina todas as faces igual.
  DIFUSA   : depende do ângulo entre a normal N e a direção da luz L.
             intensidade = max(N · L, 0)      (lei de Lambert)
             Face de frente para a luz = clara; de costas = escura.
  ESPECULAR: o brilho/reflexo. Depende também da posição do observador
             e do "brilho" do material (shininess).

Cor final = ambiente + difusa + especular
"""
import numpy as np
from OpenGL.GL import (GL_AMBIENT, GL_DIFFUSE, GL_FRONT_AND_BACK, GL_LIGHT0,
                       GL_LIGHT_MODEL_AMBIENT, GL_LIGHT_MODEL_LOCAL_VIEWER,
                       GL_LIGHTING, GL_LINES, GL_NORMALIZE, GL_POSITION,
                       GL_SHININESS, GL_SPECULAR, GL_TRUE, glBegin, glColor3f,
                       glDisable, glEnable, glEnd, glLightfv, glLightModelfv,
                       glLightModeli, glLineWidth, glMaterialf, glMaterialfv,
                       glPopMatrix, glPushMatrix, glTranslatef, glVertex3fv)
from OpenGL.GLU import gluNewQuadric, gluSphere

# Cores que a luz pode ter (tecla 7)
CORES_DA_LUZ = [
    ("Branca", (1.0, 1.0, 1.0)),
    ("Quente", (1.0, 0.78, 0.5)),
    ("Fria", (0.55, 0.75, 1.0)),
    ("Verde", (0.5, 1.0, 0.55)),
]


def criar_luz():
    """Estado inicial da luz (um dicionário simples)."""
    return {
        "ligada": True,
        "posicao": [2.0, 1.3, 0.8, 1.0],  # w = 1 -> luz pontual | w = 0 -> direcional
        "indice_cor": 0,
        "ambiente": True,
        "difusa": True,
        "especular": True,
    }


def configurar_iluminacao(luz):
    """Liga o sistema de iluminação e define as 3 componentes da GL_LIGHT0."""
    if not luz["ligada"]:
        glDisable(GL_LIGHTING)
        return

    glEnable(GL_LIGHTING)   # liga o cálculo de iluminação
    glEnable(GL_LIGHT0)     # liga a lâmpada número 0
    glEnable(GL_NORMALIZE)  # mantém |N| = 1 mesmo depois da escala

    # Observador local: o brilho especular leva em conta onde a câmera está
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.04, 0.04, 0.05, 1.0])

    r, g, b = CORES_DA_LUZ[luz["indice_cor"]][1]
    desligada = [0.0, 0.0, 0.0, 1.0]

    ambiente = [0.3 * r, 0.3 * g, 0.3 * b, 1.0] if luz["ambiente"] else desligada
    difusa = [r, g, b, 1.0] if luz["difusa"] else desligada
    especular = [r, g, b, 1.0] if luz["especular"] else desligada

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambiente)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, difusa)
    glLightfv(GL_LIGHT0, GL_SPECULAR, especular)


def posicionar_luz(luz):
    """Deve ser chamada logo DEPOIS do gluLookAt: assim a luz fica parada
    no mundo e não gira junto com o objeto."""
    glLightfv(GL_LIGHT0, GL_POSITION, luz["posicao"])


def aplicar_material(cor, brilho):
    """Como o objeto reflete cada componente da luz."""
    r, g, b = cor
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.6 * r, 0.6 * g, 0.6 * b, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [r, g, b, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.65, 0.65, 0.65, 1.0])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, brilho)


def calcular_n_dot_l(normal, ponto, luz):
    """Termo da componente difusa: N · L.
    L = direção do ponto até a luz (luz pontual) ou a própria direção
    da luz (luz direcional, w = 0)."""
    x, y, z, w = luz["posicao"]
    if w == 0:
        direcao = np.array([x, y, z])
    else:
        direcao = np.array([x, y, z]) - np.array(ponto)
    direcao = direcao / np.linalg.norm(direcao)
    return float(np.dot(normal, direcao))


_esfera = None


def desenhar_luz(luz):
    """Mostra onde está a luz: uma bolinha na cor da luz e uma linha
    tracejada até o centro do mundo."""
    global _esfera
    if _esfera is None:
        _esfera = gluNewQuadric()

    x, y, z, w = luz["posicao"]
    if w == 0:  # luz direcional não tem posição: desenhamos longe, na direção dela
        tamanho = np.linalg.norm([x, y, z])
        x, y, z = 3.2 * x / tamanho, 3.2 * y / tamanho, 3.2 * z / tamanho

    cor = CORES_DA_LUZ[luz["indice_cor"]][1]
    glDisable(GL_LIGHTING)
    glColor3f(*cor)
    glPushMatrix()
    glTranslatef(x, y, z)
    gluSphere(_esfera, 0.12, 16, 12)
    glPopMatrix()

    glLineWidth(1.5)
    glColor3f(0.6 * cor[0], 0.6 * cor[1], 0.6 * cor[2])
    glBegin(GL_LINES)
    passos = 14
    for i in range(0, passos, 2):  # tracejado: desenha um pedaço sim, outro não
        t1, t2 = i / passos, (i + 1) / passos
        glVertex3fv((x * (1 - t1), y * (1 - t1), z * (1 - t1)))
        glVertex3fv((x * (1 - t2), y * (1 - t2), z * (1 - t2)))
    glEnd()
