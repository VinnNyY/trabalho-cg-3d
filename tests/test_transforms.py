"""Parte 2 — testes das matrizes homogêneas, composição e espaços de coordenadas."""
import math

import numpy as np
import pytest

from cg3d import transforms as tf


def test_translacao_move_ponto_mas_nao_direcao():
    t = tf.translation(1, 2, 3)
    assert np.allclose(tf.transform_point(t, (0, 0, 0)), [1, 2, 3])
    assert np.allclose(tf.transform_direction(t, (1, 0, 0)), [1, 0, 0])


def test_escala():
    assert np.allclose(tf.transform_point(tf.scale(2, 3, 4), (1, 1, 1)), [2, 3, 4])


@pytest.mark.parametrize(
    "rot, entrada, esperado",
    [
        (tf.rotation_x, (0, 1, 0), (0, 0, 1)),  # Y → Z
        (tf.rotation_y, (0, 0, 1), (1, 0, 0)),  # Z → X
        (tf.rotation_z, (1, 0, 0), (0, 1, 0)),  # X → Y
    ],
)
def test_rotacoes_90_graus_seguem_mao_direita(rot, entrada, esperado):
    assert np.allclose(tf.transform_point(rot(90), entrada), esperado)


@pytest.mark.parametrize("rot", [tf.rotation_x, tf.rotation_y, tf.rotation_z])
def test_rotacao_e_ortonormal(rot):
    r = rot(37)[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3))
    assert math.isclose(np.linalg.det(r), 1.0)


def test_ordem_importa():
    """M = T·R·S gira o objeto no lugar; R·T faz o objeto orbitar a origem."""
    t, r, s = tf.translation(3, 0, 0), tf.rotation_z(90), tf.scale(2, 2, 2)
    m = tf.model_matrix(t, r, s)
    # o centro do objeto (origem local) deve ir exatamente para T
    assert np.allclose(tf.transform_point(m, (0, 0, 0)), [3, 0, 0])
    # ponto (1,0,0): escala → (2,0,0); gira → (0,2,0); translada → (3,2,0)
    assert np.allclose(tf.transform_point(m, (1, 0, 0)), [3, 2, 0])
    # ordem errada: o centro "foge" para (0,3,0) — deslocamento indesejado
    errada = r @ t @ s
    assert np.allclose(tf.transform_point(errada, (0, 0, 0)), [0, 3, 0])


def test_compose_equivale_ao_produto():
    a, b, c = tf.translation(1, 0, 0), tf.rotation_y(30), tf.scale(1, 2, 1)
    assert np.allclose(tf.compose(a, b, c), a @ b @ c)


def test_matriz_normal_com_escala_nao_uniforme():
    """Com escala não uniforme, transformar a normal por M a deixa torta;
    (M⁻¹)ᵀ a mantém perpendicular à superfície."""
    m = tf.scale(1, 3, 1)
    tangente = np.array([1.0, 1.0, 0.0])
    normal = np.array([1.0, -1.0, 0.0])
    t2 = m[:3, :3] @ tangente
    assert not math.isclose(np.dot(m[:3, :3] @ normal, t2), 0, abs_tol=1e-9)
    assert math.isclose(np.dot(tf.normal_matrix(m) @ normal, t2), 0, abs_tol=1e-9)


def test_look_at_leva_olho_para_origem_e_alvo_para_menos_z():
    eye, center = (0, 0.6, 6), (0, 0, 0)
    v = tf.look_at(eye, center, (0, 1, 0))
    assert np.allclose(tf.transform_point(v, eye), [0, 0, 0])
    alvo = tf.transform_point(v, center)
    assert np.allclose(alvo[:2], [0, 0]) and alvo[2] < 0
    assert math.isclose(np.linalg.norm(alvo), np.linalg.norm(np.subtract(center, eye)))


def test_look_at_valores_de_referencia_glu():
    """Valores calculados a partir da definição de gluLookAt para
    eye=(1,2,3), center=(0,0,0), up=(0,1,0)."""
    v = tf.look_at((1, 2, 3), (0, 0, 0), (0, 1, 0))
    f = -np.array([1, 2, 3]) / math.sqrt(14)
    s = np.cross(f, [0, 1, 0]); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    assert np.allclose(v[0, :3], s) and np.allclose(v[1, :3], u) and np.allclose(v[2, :3], -f)
    assert np.allclose(v[:3, 3], [-np.dot(s, (1, 2, 3)), -np.dot(u, (1, 2, 3)), np.dot(f, (1, 2, 3))])


def test_perspectiva_mapeia_near_e_far_para_menos1_e_1():
    p = tf.perspective(60, 16 / 9, 0.5, 50)

    def ndc_z(z):
        c = p @ np.array([0, 0, z, 1.0])
        return c[2] / c[3]

    assert math.isclose(ndc_z(-0.5), -1.0, abs_tol=1e-9)
    assert math.isclose(ndc_z(-50), 1.0, abs_tol=1e-9)
    # profundidade cresce monotonicamente com a distância
    assert ndc_z(-1) < ndc_z(-5) < ndc_z(-20)


def test_perspectiva_borda_do_fov_vai_para_borda_da_tela():
    fovy, aspect, near = 45.0, 1.5, 0.1
    p = tf.perspective(fovy, aspect, near, 100)
    z = -10.0
    y_topo = -z * math.tan(math.radians(fovy / 2))
    c = p @ np.array([y_topo * aspect, y_topo, z, 1.0])
    assert np.allclose(c[:2] / c[3], [1.0, 1.0])


def test_perspectiva_igual_a_frustum_simetrico():
    top = 0.1 * math.tan(math.radians(22.5))
    assert np.allclose(tf.perspective(45, 2, 0.1, 10), tf.frustum(-2 * top, 2 * top, -top, top, 0.1, 10))


def test_frustum_rejeita_planos_invalidos():
    with pytest.raises(ValueError):
        tf.frustum(-1, 1, -1, 1, 0, 10)


def test_to_gl_e_coluna_maior():
    m = tf.translation(5, 6, 7)
    flat = tf.to_gl(m).ravel()
    assert flat.dtype == np.float32
    assert np.allclose(flat[12:15], [5, 6, 7])  # translação nos índices 12,13,14
