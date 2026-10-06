"""
TRABALHO PRÁTICO - Aplicação Interativa 3D
Computação Gráfica e Visão Computacional

Janela com pygame + OpenGL (como nas Aulas 5_0 e 6_0).
Cada parte do trabalho está em um arquivo:
    geometria.py      -> Parte 1 (vértices, faces) e normais da Parte 3
    transformacoes.py -> Parte 2 (matrizes T, R, S)
    iluminacao.py     -> Parte 3 (luz ambiente, difusa e especular)
    cores.py          -> Parte 3 (RGB, HSV e OpenCV)
    quaternios.py     -> Parte 4 (quatérnios e trackball)

Para rodar:  python src/main.py
"""
import math
import sys

import cv2
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import gluLookAt, gluPerspective
from pygame.locals import *

import cores
import geometria
import iluminacao
import quaternios
import transformacoes as tf

LARGURA, ALTURA = 1200, 780
POSICAO_CAMERA = (0.0, 0.6, 6.0)

CONTROLES = """
=============================  CONTROLES  =============================
 Mouse (arrastar)    Trackball com quatérnios      Roda: zoom
 1 / 2 / 3           Pirâmide / Octaedro / Tetraedro
 Setas, PgUp/PgDn    Translação (matriz T)
 X / Y / Z           Rotação nos eixos (Shift inverte)  (matrizes Rx Ry Rz)
 + / -               Escala (matriz S)
 ---------------------------  ILUMINAÇÃO  ----------------------------
 L                   Liga/desliga a iluminação
 4 / 5 / 6           Liga/desliga AMBIENTE / DIFUSA / ESPECULAR
 J / K               Menos / mais brilho especular
 Shift + Setas       Move a luz (Shift + PgUp/PgDn move em Z)
 M                   Luz girando em volta do objeto
 U                   Luz pontual / direcional
 7                   Cor da luz
 B                   Sombreamento detalhado / flat
 ------------------------------  CORES  -------------------------------
 Q/A  W/S  E/D       Mais/menos R, G, B
 H                   Muda a matiz (HSV)
 O                   Usa a cor da máscara do OpenCV
 [ / ]               Muda a faixa de cor da máscara
 V                   Abre a janela do OpenCV
 -----------------------------  OUTROS  -------------------------------
 N  F  G  I          Normais | Wireframe | Eixos | Painel de texto
 Espaço              Rotação automática    R  reinicia    Esc  sai
=======================================================================
"""


# ==========================================================================
# Estado do programa (tudo o que muda enquanto o programa roda)
# ==========================================================================
def estado_inicial():
    return {
        "solido": 0,                     # índice em geometria.SOLIDOS
        "translacao": [0.0, 0.0, 0.0],   # T
        "rotacao_eixos": [0.0, 0.0, 0.0],  # ângulos para Rx, Ry, Rz
        "escala": 1.0,                   # S
        "orientacao": quaternios.IDENTIDADE,  # rotação do trackball
        "mouse_anterior": None,          # ponto na esfera no último movimento
        "distancia_camera": POSICAO_CAMERA[2],
        "cor": [0.15, 0.45, 0.95],       # cor do material (RGB)
        "brilho": 60.0,                  # shininess
        "usar_cor_opencv": False,
        "h_min": 100,                    # faixa da máscara HSV
        "h_max": 130,
        "mostrar_normais": False,
        "wireframe": False,
        "mostrar_eixos": False,
        "mostrar_painel": True,
        "detalhado": True,
        "girar_sozinho": False,
        "girar_luz": False,
        "janela_opencv": False,
    }


estado = estado_inicial()
luz = iluminacao.criar_luz()

# Imagem usada na análise de cor do OpenCV
imagem_opencv = cores.criar_imagem_paleta()
mascara = None
cor_opencv = None

# Normais e faces detalhadas são calculadas uma vez só, no início
NORMAIS = []
FACES_DETALHADAS = []
for nome, vertices, faces in geometria.SOLIDOS:
    NORMAIS.append(geometria.calcular_normais(vertices, faces))
    FACES_DETALHADAS.append(geometria.preparar_faces_detalhadas(vertices, faces))


def atualizar_mascara():
    global mascara, cor_opencv
    mascara = cores.criar_mascara(imagem_opencv, estado["h_min"], estado["h_max"])
    cor_opencv = cores.cor_media_da_mascara(imagem_opencv, mascara)


atualizar_mascara()


# ==========================================================================
# Matrizes da Parte 2
# ==========================================================================
def calcular_matriz_rotacao():
    """R = rotação do trackball (quatérnio) @ Rz @ Ry @ Rx (teclado)."""
    ax, ay, az = estado["rotacao_eixos"]
    rotacao_teclado = tf.matriz_rotacao_z(az) @ tf.matriz_rotacao_y(ay) @ tf.matriz_rotacao_x(ax)
    rotacao_trackball = quaternios.quaternio_para_matriz(estado["orientacao"])
    return rotacao_trackball @ rotacao_teclado


def calcular_matriz_modelo():
    T = tf.matriz_translacao(*estado["translacao"])
    R = calcular_matriz_rotacao()
    s = estado["escala"]
    S = tf.matriz_escala(s, s, s)
    return tf.matriz_modelo(T, R, S)


def cor_atual():
    if estado["usar_cor_opencv"] and cor_opencv is not None:
        return cor_opencv
    return estado["cor"]


# ==========================================================================
# Inicialização (Parte 1)
# ==========================================================================
def inicializar():
    pygame.init()
    try:
        # Tenta abrir com antisserrilhado (bordas mais suaves)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        pygame.display.set_mode((LARGURA, ALTURA), DOUBLEBUF | OPENGL)
    except pygame.error:
        # Se a placa de vídeo não suportar, abre sem
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
        pygame.display.set_mode((LARGURA, ALTURA), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Trabalho CG - Aplicação Interativa 3D")
    pygame.key.set_repeat(250, 30)  # segurar a tecla repete a ação

    glClearColor(0.07, 0.08, 0.11, 1.0)
    glEnable(GL_DEPTH_TEST)   # teste de profundidade: o que está na frente esconde o de trás
    glEnable(GL_CULL_FACE)    # não desenha as faces de trás (faces anti-horárias = frente)

    # Projeção em perspectiva (Aula 4_0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, LARGURA / ALTURA, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    print(CONTROLES)


# ==========================================================================
# Desenho
# ==========================================================================
def desenhar_cena():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # 1) Câmera: Espaço do Mundo -> Espaço da Câmera (Matriz de Visão)
    x, y, _ = POSICAO_CAMERA
    gluLookAt(x, y, estado["distancia_camera"], 0, 0, 0, 0, 1, 0)

    # 2) Luz, logo depois da câmera, para ficar parada no mundo
    iluminacao.configurar_iluminacao(luz)
    if luz["ligada"]:
        iluminacao.posicionar_luz(luz)
        iluminacao.desenhar_luz(luz)
    if estado["mostrar_eixos"]:
        glDisable(GL_LIGHTING)
        geometria.desenhar_eixos()

    # 3) Objeto: Espaço do Objeto -> Espaço do Mundo (Matriz de Modelo)
    glPushMatrix()
    tf.enviar_para_opengl(calcular_matriz_modelo())

    nome, vertices, faces = geometria.SOLIDOS[estado["solido"]]
    normais = NORMAIS[estado["solido"]]
    iluminacao.configurar_iluminacao(luz)
    iluminacao.aplicar_material(cor_atual(), estado["brilho"])

    if estado["wireframe"]:
        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glColor3f(*cor_atual())
        geometria.desenhar_solido(vertices, faces, normais)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_CULL_FACE)
    elif not luz["ligada"]:
        glColor3f(*cor_atual())
        geometria.desenhar_solido(vertices, faces, normais)
    elif estado["detalhado"]:
        glShadeModel(GL_SMOOTH)
        geometria.desenhar_solido_detalhado(FACES_DETALHADAS[estado["solido"]], normais)
    else:
        glShadeModel(GL_FLAT)
        geometria.desenhar_solido(vertices, faces, normais)

    if estado["mostrar_normais"]:
        glDisable(GL_LIGHTING)
        geometria.desenhar_normais(vertices, faces, normais)
    glPopMatrix()

    if estado["mostrar_painel"]:
        desenhar_painel()


# ==========================================================================
# Painel de texto na tela
# ==========================================================================
fonte = None


def escrever(texto, x, y, cor=(235, 235, 240)):
    """O pygame transforma o texto em imagem e o glDrawPixels desenha
    essa imagem na posição (x, y) da janela (origem embaixo à esquerda)."""
    superficie = fonte.render(texto, True, cor)
    dados = pygame.image.tostring(superficie, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(superficie.get_width(), superficie.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, dados)


def desenhar_painel():
    global fonte
    if fonte is None:
        fonte = pygame.font.Font(None, 22)

    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def on(valor):
        return "ON" if valor else "off"

    nome_cor, _ = iluminacao.CORES_DA_LUZ[luz["indice_cor"]]
    tipo = "pontual (w=1)" if luz["posicao"][3] == 1 else "direcional (w=0)"
    r, g, b = cor_atual()
    h, s, v = cores.rgb_para_hsv(r, g, b)
    px, py, pz, _ = luz["posicao"]

    linhas = [
        ("ILUMINAÇÃO - modelo de Phong (Parte 3)", (255, 255, 255)),
        (f"Luz {on(luz['ligada'])}: {tipo}, posição ({px:.1f}, {py:.1f}, {pz:.1f}), cor {nome_cor}", (170, 175, 190)),
        (f"[4] Ambiente {on(luz['ambiente'])}", (120, 200, 255)),
        (f"[5] Difusa {on(luz['difusa'])}", (255, 210, 110)),
        (f"[6] Especular {on(luz['especular'])}   brilho = {estado['brilho']:.0f}  [J/K]", (255, 255, 255)),
        (f"Cor RGB ({r:.2f}, {g:.2f}, {b:.2f})   HSV ({h:.0f}°, {s:.2f}, {v:.2f})", (170, 175, 190)),
        (f"[B] Sombreamento: {'detalhado' if estado['detalhado'] else 'flat'}   [M] luz girando: {on(estado['girar_luz'])}", (170, 175, 190)),
    ]
    y = ALTURA - 28
    for texto, cor in linhas:
        escrever(texto, 16, y, cor)
        y -= 22

    # N · L de cada face: quanto cada face está virada para a luz
    if luz["ligada"]:
        nome, vertices, faces = geometria.SOLIDOS[estado["solido"]]
        normais = NORMAIS[estado["solido"]]
        M = calcular_matriz_modelo()
        R = calcular_matriz_rotacao()
        y = 16 + 22 * (len(faces) + 1)
        escrever(f"{nome}: N · L de cada face (difusa = max(N·L, 0))", LARGURA - 400, y)
        for i, face in enumerate(faces):
            y -= 22
            normal_mundo = R[:3, :3] @ normais[i]  # normal girada junto com o objeto
            centro_mundo = tf.transformar_ponto(M, geometria.centro_da_face(vertices, face))
            n_dot_l = iluminacao.calcular_n_dot_l(normal_mundo, centro_mundo, luz)
            if n_dot_l > 0:
                texto, cor = f"face {i}:  N·L = {n_dot_l:+.2f}   iluminada", (255, 210, 110)
            else:
                texto, cor = f"face {i}:  N·L = {n_dot_l:+.2f}   de costas (só ambiente)", (120, 125, 140)
            escrever(texto, LARGURA - 400, y, cor)

    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)


# ==========================================================================
# Teclado
# ==========================================================================
def tratar_tecla(tecla):
    shift = pygame.key.get_mods() & KMOD_SHIFT

    # --- Sólido -----------------------------------------------------------
    if tecla in (K_1, K_2, K_3):
        estado["solido"] = tecla - K_1

    # --- Translação (ou luz, com Shift) ------------------------------------
    elif tecla in (K_LEFT, K_RIGHT, K_UP, K_DOWN, K_PAGEUP, K_PAGEDOWN):
        dx = {K_LEFT: -1, K_RIGHT: 1}.get(tecla, 0)
        dy = {K_DOWN: -1, K_UP: 1}.get(tecla, 0)
        dz = {K_PAGEUP: -1, K_PAGEDOWN: 1}.get(tecla, 0)
        if shift:
            luz["posicao"][0] += dx * 0.3
            luz["posicao"][1] += dy * 0.3
            luz["posicao"][2] += dz * 0.3
        else:
            estado["translacao"][0] += dx * 0.1
            estado["translacao"][1] += dy * 0.1
            estado["translacao"][2] += dz * 0.1

    # --- Rotação nos eixos ---------------------------------------------------
    elif tecla in (K_x, K_y, K_z):
        eixo = {K_x: 0, K_y: 1, K_z: 2}[tecla]
        estado["rotacao_eixos"][eixo] += -5 if shift else 5

    # --- Escala ----------------------------------------------------------------
    elif tecla in (K_EQUALS, K_PLUS, K_KP_PLUS):
        estado["escala"] = min(estado["escala"] * 1.1, 4.0)
    elif tecla in (K_MINUS, K_KP_MINUS):
        estado["escala"] = max(estado["escala"] / 1.1, 0.2)

    # --- Iluminação ------------------------------------------------------------
    elif tecla == K_l:
        luz["ligada"] = not luz["ligada"]
    elif tecla == K_4:
        luz["ambiente"] = not luz["ambiente"]
    elif tecla == K_5:
        luz["difusa"] = not luz["difusa"]
    elif tecla == K_6:
        luz["especular"] = not luz["especular"]
    elif tecla == K_j:
        estado["brilho"] = max(estado["brilho"] / 1.4, 1)
    elif tecla == K_k:
        estado["brilho"] = min(estado["brilho"] * 1.4, 128)
    elif tecla == K_m:
        estado["girar_luz"] = not estado["girar_luz"]
    elif tecla == K_u:
        luz["posicao"][3] = 0.0 if luz["posicao"][3] == 1.0 else 1.0
    elif tecla == K_7:
        luz["indice_cor"] = (luz["indice_cor"] + 1) % len(iluminacao.CORES_DA_LUZ)
    elif tecla == K_b:
        estado["detalhado"] = not estado["detalhado"]

    # --- Cores RGB / HSV ---------------------------------------------------------
    elif tecla in (K_q, K_a, K_w, K_s, K_e, K_d):
        canal = {K_q: 0, K_a: 0, K_w: 1, K_s: 1, K_e: 2, K_d: 2}[tecla]
        passo = 0.05 if tecla in (K_q, K_w, K_e) else -0.05
        estado["cor"][canal] = min(max(estado["cor"][canal] + passo, 0.0), 1.0)
        estado["usar_cor_opencv"] = False
        mostrar_conversao_de_cor()
    elif tecla == K_h:
        h, s, v = cores.rgb_para_hsv(*estado["cor"])
        h = (h + 20) % 360           # gira a matiz 20 graus no círculo de cores
        estado["cor"] = list(cores.hsv_para_rgb(h, max(s, 0.5), max(v, 0.3)))
        estado["usar_cor_opencv"] = False
        mostrar_conversao_de_cor()

    # --- OpenCV ----------------------------------------------------------------
    elif tecla == K_o:
        estado["usar_cor_opencv"] = not estado["usar_cor_opencv"]
        mostrar_conversao_de_cor()
    elif tecla in (K_LEFTBRACKET, K_RIGHTBRACKET):
        passo = -5 if tecla == K_LEFTBRACKET else 5
        largura_faixa = estado["h_max"] - estado["h_min"]
        estado["h_min"] = min(max(estado["h_min"] + passo, 0), 179 - largura_faixa)
        estado["h_max"] = estado["h_min"] + largura_faixa
        atualizar_mascara()
        estado["usar_cor_opencv"] = True
        print(f"Máscara HSV: H de {estado['h_min']} a {estado['h_max']}")
        mostrar_conversao_de_cor()
    elif tecla == K_v:
        estado["janela_opencv"] = not estado["janela_opencv"]
        if not estado["janela_opencv"]:
            cv2.destroyAllWindows()

    # --- Outros ----------------------------------------------------------------
    elif tecla == K_n:
        estado["mostrar_normais"] = not estado["mostrar_normais"]
    elif tecla == K_f:
        estado["wireframe"] = not estado["wireframe"]
    elif tecla == K_g:
        estado["mostrar_eixos"] = not estado["mostrar_eixos"]
    elif tecla == K_i:
        estado["mostrar_painel"] = not estado["mostrar_painel"]
    elif tecla == K_SPACE:
        estado["girar_sozinho"] = not estado["girar_sozinho"]
    elif tecla == K_r:
        reiniciar()


def mostrar_conversao_de_cor():
    """Mostra no terminal a nossa conversão RGB -> HSV ao lado da do OpenCV."""
    r, g, b = cor_atual()
    h, s, v = cores.rgb_para_hsv(r, g, b)
    h_cv, s_cv, v_cv = cores.rgb_para_hsv_opencv(r, g, b)
    print(f"RGB ({r:.2f}, {g:.2f}, {b:.2f}) -> HSV ({h:.0f}°, {s:.2f}, {v:.2f}) | "
          f"escala OpenCV: nossa ({h / 2:.0f}, {s * 255:.0f}, {v * 255:.0f})  "
          f"cv2 ({h_cv}, {s_cv}, {v_cv})")


def reiniciar():
    global estado, luz
    solido = estado["solido"]
    estado = estado_inicial()
    estado["solido"] = solido
    luz = iluminacao.criar_luz()
    atualizar_mascara()


# ==========================================================================
# Mouse - Trackball (Parte 4)
# ==========================================================================
def tratar_mouse(evento):
    if evento.type == MOUSEBUTTONDOWN and evento.button == 1:
        # Clique: guarda o ponto inicial sobre a esfera
        estado["mouse_anterior"] = quaternios.mapear_para_esfera(*evento.pos, LARGURA, ALTURA)

    elif evento.type == MOUSEBUTTONUP and evento.button == 1:
        estado["mouse_anterior"] = None

    elif evento.type == MOUSEMOTION and estado["mouse_anterior"] is not None:
        # Arraste: rotação entre o ponto anterior e o atual
        ponto_atual = quaternios.mapear_para_esfera(*evento.pos, LARGURA, ALTURA)
        q_arraste = quaternios.rotacao_do_arraste(estado["mouse_anterior"], ponto_atual)
        # Acumula: nova orientação = rotação do arraste * orientação anterior
        estado["orientacao"] = quaternios.normalizar(
            quaternios.multiplicar(q_arraste, estado["orientacao"]))
        estado["mouse_anterior"] = ponto_atual

    elif evento.type == MOUSEWHEEL:
        estado["distancia_camera"] = min(max(estado["distancia_camera"] - evento.y * 0.4, 2.5), 25)


# ==========================================================================
# Animações
# ==========================================================================
def atualizar(dt):
    if estado["girar_sozinho"] and estado["mouse_anterior"] is None:
        passo = quaternios.quaternio_de_eixo_angulo((0, 1, 0), math.radians(45) * dt)
        estado["orientacao"] = quaternios.normalizar(quaternios.multiplicar(passo, estado["orientacao"]))

    if estado["girar_luz"]:
        # Gira a posição da luz em torno do eixo Y usando a matriz da Parte 2
        x, y, z, w = luz["posicao"]
        nova = tf.matriz_rotacao_y(60 * dt) @ np.array([x, y, z, 0], dtype=np.float32)
        luz["posicao"] = [float(nova[0]), float(nova[1]), float(nova[2]), w]

    if estado["janela_opencv"]:
        cv2.imshow("Analise HSV (OpenCV)", cores.montar_painel(imagem_opencv, mascara, cor_opencv))
        cv2.waitKey(1)


# ==========================================================================
# Laço principal
# ==========================================================================
def main():
    global imagem_opencv
    if len(sys.argv) > 2 and sys.argv[1] == "--imagem":
        imagem = cv2.imread(sys.argv[2])
        if imagem is not None:
            imagem_opencv = cv2.resize(imagem, (360, int(360 * imagem.shape[0] / imagem.shape[1])))
            atualizar_mascara()

    inicializar()
    relogio = pygame.time.Clock()

    while True:
        dt = relogio.tick(60) / 1000  # segundos desde o último quadro

        for evento in pygame.event.get():
            if evento.type == QUIT or (evento.type == KEYDOWN and evento.key == K_ESCAPE):
                cv2.destroyAllWindows()
                pygame.quit()
                return
            if evento.type == KEYDOWN:
                tratar_tecla(evento.key)
            tratar_mouse(evento)

        atualizar(dt)
        desenhar_cena()
        pygame.display.flip()


if __name__ == "__main__":
    main()
