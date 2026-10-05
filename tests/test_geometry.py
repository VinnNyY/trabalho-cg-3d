"""Partes 1 e 3 — malhas e vetores normais."""
import math

import numpy as np
import pytest

from cg3d.geometry import all_meshes, face_normal, octahedron, pyramid, tetrahedron

MESHES = all_meshes()
IDS = [m.name for m in MESHES]


def test_normal_pelo_produto_vetorial():
    # triângulo no plano XY, anti-horário visto de +Z → normal +Z
    assert np.allclose(face_normal((0, 0, 0), (1, 0, 0), (0, 1, 0)), [0, 0, 1])
    # invertendo a ordem a normal inverte
    assert np.allclose(face_normal((0, 0, 0), (0, 1, 0), (1, 0, 0)), [0, 0, -1])


def test_face_degenerada_gera_erro():
    with pytest.raises(ValueError):
        face_normal((0, 0, 0), (1, 1, 1), (2, 2, 2))


@pytest.mark.parametrize("mesh", MESHES, ids=IDS)
def test_normais_unitarias(mesh):
    assert np.allclose(np.linalg.norm(mesh.face_normals, axis=1), 1.0)


@pytest.mark.parametrize("mesh", MESHES, ids=IDS)
def test_normais_apontam_para_fora(mesh):
    """Garante a ordem anti-horária de todas as faces (vital para o culling
    e para a iluminação)."""
    c = mesh.centroid()
    for i in range(len(mesh.faces)):
        assert np.dot(mesh.face_normals[i], mesh.face_centroid(i) - c) > 0, f"face {i} invertida"


@pytest.mark.parametrize("mesh", MESHES, ids=IDS)
def test_faces_planas(mesh):
    for i, f in enumerate(mesh.faces):
        p0 = mesh.vertices[f[0]]
        for idx in f[3:]:
            assert math.isclose(np.dot(mesh.vertices[idx] - p0, mesh.face_normals[i]), 0, abs_tol=1e-9)


@pytest.mark.parametrize("mesh", MESHES, ids=IDS)
def test_euler_poliedro_fechado(mesh):
    assert mesh.euler_characteristic() == 2


@pytest.mark.parametrize("mesh", MESHES, ids=IDS)
def test_malha_fechada_cada_aresta_em_duas_faces_opostas(mesh):
    """Cada aresta orientada (a→b) aparece uma única vez e sua inversa (b→a)
    também — característica de malha fechada e orientada de forma consistente."""
    directed = []
    for f in mesh.faces:
        directed += list(zip(f, f[1:] + f[:1]))
    assert len(directed) == len(set(directed))
    assert all((b, a) in set(directed) for a, b in directed)


def test_contagens():
    assert (len(pyramid().vertices), len(pyramid().faces)) == (5, 5)
    assert len(pyramid().quads) == 1 and len(pyramid().triangles) == 4
    assert (len(octahedron().vertices), len(octahedron().faces)) == (6, 8)
    assert (len(tetrahedron().vertices), len(tetrahedron().faces)) == (4, 4)


def test_tetraedro_regular():
    v = tetrahedron().vertices
    d = {round(float(np.linalg.norm(v[i] - v[j])), 9) for i in range(4) for j in range(i + 1, 4)}
    assert len(d) == 1


def test_piramide_centrada_no_centroide_de_volume():
    p = pyramid()
    ys = p.vertices[:, 1]
    altura = ys.max() - ys.min()
    assert math.isclose(-ys.min(), altura / 4)
