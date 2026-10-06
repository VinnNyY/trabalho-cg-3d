"""
Testes da parte matemática do trabalho.  Para rodar:  pytest

Cada teste confere uma propriedade que vimos em aula.
"""
import math

import numpy as np

import cores
import geometria
import quaternios
import transformacoes as tf


# ---------------------------------------------------------------- Parte 1
def test_piramide_tem_5_vertices_e_5_faces():
    assert len(geometria.PIRAMIDE_VERTICES) == 5
    assert len(geometria.PIRAMIDE_FACES) == 5


def test_todas_as_normais_apontam_para_fora():
    """Se a face está em sentido anti-horário, a normal aponta para fora:
    o produto escalar entre a normal e o vetor (centro do sólido -> centro da
    face) é positivo."""
    for nome, vertices, faces in geometria.SOLIDOS:
        centro_solido = np.mean(np.array(vertices), axis=0)
        normais = geometria.calcular_normais(vertices, faces)
        for i, face in enumerate(faces):
            centro_face = geometria.centro_da_face(vertices, face)
            assert np.dot(normais[i], centro_face - centro_solido) > 0, f"{nome}, face {i}"


# ---------------------------------------------------------------- Parte 2
def test_translacao():
    T = tf.matriz_translacao(1, 2, 3)
    assert np.allclose(tf.transformar_ponto(T, (0, 0, 0)), (1, 2, 3))


def test_escala():
    S = tf.matriz_escala(2, 3, 4)
    assert np.allclose(tf.transformar_ponto(S, (1, 1, 1)), (2, 3, 4))


def test_rotacoes_de_90_graus():
    assert np.allclose(tf.transformar_ponto(tf.matriz_rotacao_x(90), (0, 1, 0)), (0, 0, 1), atol=1e-6)
    assert np.allclose(tf.transformar_ponto(tf.matriz_rotacao_y(90), (0, 0, 1)), (1, 0, 0), atol=1e-6)
    assert np.allclose(tf.transformar_ponto(tf.matriz_rotacao_z(90), (1, 0, 0)), (0, 1, 0), atol=1e-6)


def test_ordem_das_transformacoes_importa():
    T = tf.matriz_translacao(3, 0, 0)
    R = tf.matriz_rotacao_z(90)
    S = tf.matriz_escala(2, 2, 2)
    # Ordem correta (T @ R @ S): o centro do objeto vai exatamente para (3, 0, 0)
    M = tf.matriz_modelo(T, R, S)
    assert np.allclose(tf.transformar_ponto(M, (0, 0, 0)), (3, 0, 0))
    # Ordem errada (R @ T @ S): o centro "orbita" e vai parar em (0, 3, 0)
    errada = R @ T @ S
    assert np.allclose(tf.transformar_ponto(errada, (0, 0, 0)), (0, 3, 0), atol=1e-6)


# ---------------------------------------------------------------- Parte 3
def test_normal_pelo_produto_vetorial():
    normal = geometria.calcular_normal((0, 0, 0), (1, 0, 0), (0, 1, 0))
    assert np.allclose(normal, (0, 0, 1))


def test_normais_tem_tamanho_1():
    for nome, vertices, faces in geometria.SOLIDOS:
        for normal in geometria.calcular_normais(vertices, faces):
            assert math.isclose(np.linalg.norm(normal), 1.0)


def test_divisao_da_face_gera_4_triangulos_por_nivel():
    a, b, c = np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([0, 1, 0.0])
    assert len(geometria.dividir_triangulo(a, b, c, 1)) == 4
    assert len(geometria.dividir_triangulo(a, b, c, 3)) == 64


def test_rgb_para_hsv():
    assert cores.rgb_para_hsv(1, 0, 0) == (0, 1, 1)        # vermelho: H = 0°
    assert cores.rgb_para_hsv(0, 1, 0) == (120, 1, 1)      # verde:    H = 120°
    assert cores.rgb_para_hsv(0, 0, 1) == (240, 1, 1)      # azul:     H = 240°


def test_hsv_ida_e_volta():
    for cor in [(0.2, 0.6, 0.9), (0.95, 0.4, 0.1), (0.5, 0.5, 0.5)]:
        assert np.allclose(cores.hsv_para_rgb(*cores.rgb_para_hsv(*cor)), cor)


def test_nossa_conversao_bate_com_opencv():
    r, g, b = 0.2, 0.6, 0.9
    h, s, v = cores.rgb_para_hsv(r, g, b)
    h_cv, s_cv, v_cv = cores.rgb_para_hsv_opencv(r, g, b)
    assert abs(h / 2 - h_cv) <= 1 and abs(s * 255 - s_cv) <= 1 and abs(v * 255 - v_cv) <= 1


def test_mascara_azul_gera_cor_azulada():
    imagem = cores.criar_imagem_paleta()
    mascara = cores.criar_mascara(imagem, 100, 130)
    r, g, b = cores.cor_media_da_mascara(imagem, mascara)
    assert b > r and b > g


# ---------------------------------------------------------------- Parte 4
def test_quaternio_igual_a_matriz_de_rotacao():
    """Quatérnio de 90° em Z gera a mesma matriz que matriz_rotacao_z(90)."""
    q = quaternios.quaternio_de_eixo_angulo((0, 0, 1), math.radians(90))
    assert np.allclose(quaternios.quaternio_para_matriz(q), tf.matriz_rotacao_z(90), atol=1e-6)


def test_quaternio_e_unitario():
    q = quaternios.quaternio_de_eixo_angulo((1, 2, 3), 1.0)
    assert math.isclose(sum(c * c for c in q), 1.0)


def test_multiplicar_quaternios_igual_multiplicar_matrizes():
    q1 = quaternios.quaternio_de_eixo_angulo((1, 0, 0), 0.7)
    q2 = quaternios.quaternio_de_eixo_angulo((0, 1, 0), -1.1)
    m_quaternio = quaternios.quaternio_para_matriz(quaternios.multiplicar(q2, q1))
    m_matrizes = quaternios.quaternio_para_matriz(q2) @ quaternios.quaternio_para_matriz(q1)
    assert np.allclose(m_quaternio, m_matrizes, atol=1e-6)


def test_centro_da_tela_vai_para_o_topo_da_esfera():
    assert np.allclose(quaternios.mapear_para_esfera(400, 300, 800, 600), (0, 0, 1))


def test_ponto_do_mouse_fica_sobre_a_esfera():
    for x, y in [(100, 100), (400, 50), (790, 590)]:
        ponto = quaternios.mapear_para_esfera(x, y, 800, 600)
        assert math.isclose(np.linalg.norm(ponto), 1.0)


def test_arrastar_para_direita_gira_em_torno_de_y():
    p1 = quaternios.mapear_para_esfera(400, 300, 800, 600)
    p2 = quaternios.mapear_para_esfera(500, 300, 800, 600)
    w, x, y, z = quaternios.rotacao_do_arraste(p1, p2)
    assert y > 0 and math.isclose(x, 0, abs_tol=1e-9) and math.isclose(z, 0, abs_tol=1e-9)


def test_sem_gimbal_lock():
    """Com ângulos de Euler, depois de girar 90° em Y, girar em X ou em Z dá
    o MESMO resultado (perdemos um eixo = Gimbal Lock).
    Com quatérnios, as duas rotações continuam diferentes."""
    ry = tf.matriz_rotacao_y(90)
    assert np.allclose(ry @ tf.matriz_rotacao_x(25), tf.matriz_rotacao_z(-25) @ ry, atol=1e-6)

    q = quaternios.quaternio_de_eixo_angulo((0, 1, 0), math.radians(90))
    qx = quaternios.quaternio_de_eixo_angulo((1, 0, 0), math.radians(25))
    qz = quaternios.quaternio_de_eixo_angulo((0, 0, 1), math.radians(-25))
    assert not np.allclose(quaternios.quaternio_para_matriz(quaternios.multiplicar(qx, q)),
                           quaternios.quaternio_para_matriz(quaternios.multiplicar(qz, q)))
