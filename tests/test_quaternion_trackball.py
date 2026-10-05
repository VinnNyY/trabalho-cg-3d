"""Parte 4 — quatérnios e trackball virtual."""
import math

import numpy as np
import pytest

from cg3d import transforms as tf
from cg3d.quaternion import Quaternion
from cg3d.trackball import Trackball, project_to_sphere, rotation_between, screen_to_ndc


@pytest.mark.parametrize(
    "axis, rot",
    [((1, 0, 0), tf.rotation_x), ((0, 1, 0), tf.rotation_y), ((0, 0, 1), tf.rotation_z)],
)
@pytest.mark.parametrize("deg", [0, 30, 90, 180, 270])
def test_quaternio_para_matriz_igual_rotacoes_da_parte2(axis, rot, deg):
    q = Quaternion.from_axis_angle(axis, math.radians(deg))
    assert np.allclose(q.to_matrix(), rot(deg))


def test_quaternio_unitario():
    q = Quaternion.from_axis_angle((1, 2, 3), 1.234)
    assert math.isclose(q.norm(), 1.0)


def test_produto_de_hamilton_i_j_k():
    i, j, k = Quaternion(0, 1, 0, 0), Quaternion(0, 0, 1, 0), Quaternion(0, 0, 0, 1)
    assert (i * j).as_tuple() == (0, 0, 0, 1)            # ij = k
    assert (j * i).as_tuple() == (0, 0, 0, -1)           # ji = -k (não comuta)
    assert (i * i).as_tuple() == (-1, 0, 0, 0)           # i² = -1
    assert np.allclose((i * j * k).as_tuple(), (-1, 0, 0, 0))  # ijk = -1


def test_composicao_de_quaternios_igual_produto_de_matrizes():
    q1 = Quaternion.from_axis_angle((1, 1, 0), 0.7)
    q2 = Quaternion.from_axis_angle((0, 1, 1), -1.1)
    assert np.allclose((q2 * q1).to_matrix(), q2.to_matrix() @ q1.to_matrix())


def test_rotate_vetor_igual_matriz():
    q = Quaternion.from_axis_angle((0.3, -0.5, 0.8), 2.0)
    v = np.array([1.0, 2.0, -0.5])
    assert np.allclose(q.rotate(v), q.to_matrix()[:3, :3] @ v)


def test_conjugado_desfaz_rotacao():
    q = Quaternion.from_axis_angle((1, 2, 3), 0.9)
    assert np.allclose((q * q.conjugate()).as_tuple(), (1, 0, 0, 0))


def test_eixo_angulo_ida_e_volta():
    axis = np.array([2.0, -1.0, 0.5]); axis /= np.linalg.norm(axis)
    a2, ang = Quaternion.from_axis_angle(axis, 1.3).to_axis_angle()
    assert np.allclose(a2, axis) and math.isclose(ang, 1.3)


def test_sem_gimbal_lock():
    """Com Euler, girar 90° em Y faz X e Z produzirem a MESMA rotação
    (perde-se um grau de liberdade). Com quatérnios, o incremento no eixo X
    da câmera continua sendo uma rotação em X, qualquer que seja a orientação."""
    pitch = tf.rotation_y(90)
    # Euler Rz·Ry·Rx: com Ry=90°, Rx(a) e Rz(-a) dão o mesmo resultado → lock
    a = 25
    assert np.allclose(tf.rotation_z(0) @ pitch @ tf.rotation_x(a),
                       tf.rotation_z(-a) @ pitch @ tf.rotation_x(0))
    # Quatérnio: orientação atual = 90° em Y; aplica 25° em X (eixo de tela)
    q = Quaternion.from_axis_angle((0, 1, 0), math.pi / 2)
    q_x = Quaternion.from_axis_angle((1, 0, 0), math.radians(a))
    q_z = Quaternion.from_axis_angle((0, 0, 1), math.radians(-a))
    assert not np.allclose((q_x * q).to_matrix(), (q_z * q).to_matrix())


def test_renormalizacao_mantem_norma_apos_muitas_composicoes():
    q = Quaternion.identity()
    step = Quaternion.from_axis_angle((0.2, 0.9, 0.4), 0.0137)
    for _ in range(100_000):
        q = (step * q).normalized()
    assert math.isclose(q.norm(), 1.0, abs_tol=1e-12)
    r = q.to_matrix()[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3))


# ------------------------------------------------------------------ #
# Trackball
# ------------------------------------------------------------------ #
def test_centro_da_tela_vai_para_o_polo():
    assert np.allclose(project_to_sphere(*screen_to_ndc(400, 300, 800, 600)), [0, 0, 1])


def test_y_da_tela_e_invertido():
    nx, ny = screen_to_ndc(400, 0, 800, 600)  # topo da janela
    assert math.isclose(nx, 0) and ny > 0


@pytest.mark.parametrize("p", [(0.3, 0.4), (-0.9, 0.1), (0.0, 0.0), (1.5, -2.0), (5, 5)])
def test_projecao_e_unitaria_e_no_hemisferio_frontal(p):
    v = project_to_sphere(*p)
    assert math.isclose(np.linalg.norm(v), 1.0) and v[2] >= 0


def test_rotation_between_leva_p0_em_p1():
    p0, p1 = project_to_sphere(0.1, 0.2), project_to_sphere(-0.4, 0.5)
    q = rotation_between(p0, p1)
    assert np.allclose(q.rotate(p0), p1)


def test_arrastar_para_direita_gira_em_torno_de_y_positivo():
    tb = Trackball(sensitivity=1.0)
    tb.begin(400, 300, 800, 600)
    tb.drag(500, 300, 800, 600)
    axis, ang = tb.orientation.to_axis_angle()
    assert np.allclose(axis, [0, 1, 0]) and ang > 0


def test_arrastar_para_cima_gira_em_torno_de_x_negativo():
    tb = Trackball(sensitivity=1.0)
    tb.begin(400, 300, 800, 600)
    tb.drag(400, 200, 800, 600)
    axis, _ = tb.orientation.to_axis_angle()
    assert np.allclose(axis, [-1, 0, 0])


def test_arraste_acumula_e_ida_e_volta_retorna_a_identidade():
    tb = Trackball(sensitivity=1.0)
    tb.begin(400, 300, 800, 600)
    for x in range(400, 560, 8):
        tb.drag(x, 300, 800, 600)
    assert not np.allclose(tb.matrix(), np.eye(4))
    for x in range(552, 392, -8):
        tb.drag(x, 300, 800, 600)
    assert np.allclose(tb.matrix(), np.eye(4), atol=1e-9)
    tb.end()
    assert not tb.dragging


def test_reset():
    tb = Trackball()
    tb.begin(0, 0, 100, 100)
    tb.drag(60, 40, 100, 100)
    tb.reset()
    assert tb.orientation.as_tuple() == (1, 0, 0, 0)
