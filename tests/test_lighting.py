"""Parte 3 — equação de iluminação (ambiente + difusa + especular)."""
import math

import numpy as np
import pytest

from cg3d.geometry import all_meshes, subdivide_faces
from cg3d.lighting import GLOBAL_AMBIENT, Light, Material, light_direction, phong_shade

EYE = (0.0, 0.0, 5.0)


def luz_em(x, y, z, **kw):
    return Light(position=[x, y, z, 1.0], **kw)


def test_face_de_frente_para_a_luz_recebe_difusa_maxima():
    lt, mat = luz_em(0, 0, 5), Material()
    r = phong_shade((0, 0, 1), (0, 0, 0), EYE, lt, mat)
    assert math.isclose(r.n_dot_l, 1.0)
    assert np.allclose(r.diffuse, np.array(mat.color) * lt.diffuse_intensity)


def test_lei_do_cosseno_de_lambert():
    """Difusa ∝ cos θ: luz a 60° da normal → metade da difusa."""
    mat = Material()
    frente = phong_shade((0, 0, 1), (0, 0, 0), EYE, Light(position=[0, 0, 1, 0.0]), mat)
    a60 = phong_shade((0, 0, 1), (0, 0, 0), EYE,
                      Light(position=[math.sin(math.radians(60)), 0, math.cos(math.radians(60)), 0.0]), mat)
    assert math.isclose(a60.n_dot_l, 0.5, abs_tol=1e-12)
    assert np.allclose(a60.diffuse, frente.diffuse * 0.5)


def test_face_de_costas_so_recebe_ambiente():
    lt, mat = luz_em(0, 0, -5), Material()
    r = phong_shade((0, 0, 1), (0, 0, 0), EYE, lt, mat)
    assert r.n_dot_l < 0
    assert np.allclose(r.diffuse, 0) and np.allclose(r.specular, 0)
    esperado = np.array(GLOBAL_AMBIENT) * mat.ambient + np.array(lt.ambient[:3]) * mat.ambient
    assert np.allclose(r.color, esperado)


def test_especular_maximo_quando_h_coincide_com_n():
    """Luz e olho simétricos em relação à normal → H = N → reflexo máximo."""
    mat = Material()
    r = phong_shade((0, 0, 1), (0, 0, 0), (0, 3, 4), luz_em(0, -3, 4), mat)
    assert math.isclose(r.n_dot_h, 1.0)
    assert np.allclose(r.specular, mat.specular)


def test_brilho_maior_deixa_o_reflexo_mais_concentrado():
    fora_do_centro = dict(normal=(0, 0, 1), point=(0, 0, 0), eye=(0, 1, 4), light=luz_em(0, -3, 4))
    largo = phong_shade(material=Material(shininess=5), **fora_do_centro)
    estreito = phong_shade(material=Material(shininess=100), **fora_do_centro)
    assert 0 < largo.n_dot_h < 1
    assert estreito.specular.sum() < largo.specular.sum()


@pytest.mark.parametrize("comp", ["ambient", "diffuse", "specular"])
def test_desligar_componente_zera_a_parcela(comp):
    lt = luz_em(0, -3, 4)
    setattr(lt, f"use_{comp}", False)
    r = phong_shade((0, 0, 1), (0, 0, 0), (0, 3, 4), lt, Material())
    parcela = {"ambient": r.ambient - np.array(GLOBAL_AMBIENT) * Material().ambient,
               "diffuse": r.diffuse, "specular": r.specular}[comp]
    assert np.allclose(parcela, 0)


def test_soma_das_parcelas():
    r = phong_shade((0, 0.6, 0.8), (0, 0, 0), EYE, luz_em(1, 2, 3), Material())
    assert np.allclose(r.color, np.clip(r.ambient + r.diffuse + r.specular, 0, 1))


def test_cor_da_luz_multiplica_a_cor_do_material():
    mat = Material(color=(1.0, 1.0, 1.0))
    r = phong_shade((0, 0, 1), (0, 0, 0), EYE, Light(position=[0, 0, 1, 0.0], color_name="Fria",
                                                    use_specular=False, use_ambient=False), mat)
    assert np.allclose(r.diffuse, (0.55, 0.75, 1.00))


def test_luz_pontual_x_direcional():
    pontual = luz_em(0, 2, 0)
    direcional = Light(position=[0, 2, 0, 0.0])
    # pontual: L muda de ponto para ponto
    assert not np.allclose(light_direction(pontual, (-3, 0, 0)), light_direction(pontual, (3, 0, 0)))
    # direcional: raios paralelos, L igual em toda a cena
    assert np.allclose(light_direction(direcional, (-3, 0, 0)), light_direction(direcional, (3, 0, 0)))
    assert np.allclose(light_direction(direcional, (0, 0, 0)), (0, 1, 0))


def test_toggle_tipo_e_cor_da_luz():
    lt = Light()
    assert not lt.directional
    lt.toggle_type()
    assert lt.directional
    nome = lt.color_name
    lt.next_color()
    assert lt.color_name != nome


# ------------------------------------------------------------------ #
# Subdivisão usada no sombreamento detalhado
# ------------------------------------------------------------------ #
def _area(tri):
    return 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))


@pytest.mark.parametrize("mesh", all_meshes(), ids=lambda m: m.name)
def test_subdivisao_preserva_area_normais_e_orientacao(mesh):
    pos, nor = subdivide_faces(mesh, level=6)
    tris = pos.reshape(-1, 3, 3).astype(np.float64)
    norms = nor.reshape(-1, 3, 3)
    area_original = sum(
        sum(_area(mesh.vertices[[f[0], f[k], f[k + 1]]]) for k in range(1, len(f) - 1))
        for f in mesh.faces
    )
    assert math.isclose(sum(_area(t) for t in tris), area_original, rel_tol=1e-5)
    for t, n in zip(tris, norms):
        assert np.allclose(n[0], n[1]) and np.allclose(n[0], n[2])
        # mesma orientação anti-horária da face original
        assert np.dot(np.cross(t[1] - t[0], t[2] - t[0]), n[0]) > 0
