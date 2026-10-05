"""Parte 3 — modelos de cor e análise com OpenCV."""
import colorsys

import cv2
import numpy as np
import pytest

from cg3d import color as cm

AMOSTRAS = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1),
    (0, 0, 0), (1, 1, 1), (0.5, 0.5, 0.5), (0.2, 0.6, 0.9), (0.95, 0.4, 0.1),
]


@pytest.mark.parametrize("rgb", AMOSTRAS)
def test_rgb_para_hsv_bate_com_colorsys(rgb):
    h, s, v = cm.rgb_to_hsv(*rgb)
    h2, s2, v2 = colorsys.rgb_to_hsv(*rgb)
    assert np.isclose(h / 360, h2) and np.isclose(s, s2) and np.isclose(v, v2)


@pytest.mark.parametrize("rgb", AMOSTRAS)
def test_hsv_ida_e_volta(rgb):
    assert np.allclose(cm.hsv_to_rgb(*cm.rgb_to_hsv(*rgb)), rgb)


@pytest.mark.parametrize("rgb", AMOSTRAS)
def test_nossa_conversao_bate_com_cv2_cvtColor(rgb):
    """H do OpenCV = graus/2 e S,V em [0,255]; tolerância de 1 unidade
    pelo arredondamento para 8 bits."""
    ours = cm.hsv_degrees_to_opencv(*cm.rgb_to_hsv(*rgb))
    theirs = cm.rgb_to_hsv_opencv(*rgb)
    dh = min(abs(ours[0] - theirs[0]), 180 - abs(ours[0] - theirs[0]))
    assert dh <= 1 and abs(ours[1] - theirs[1]) <= 1 and abs(ours[2] - theirs[2]) <= 1


def test_paleta_tem_todos_os_matizes():
    hsv = cv2.cvtColor(cm.make_palette_image(), cv2.COLOR_BGR2HSV)
    top = hsv[0, :, 0]
    assert top.min() <= 2 and top.max() >= 175


def test_mascara_azul_produz_cor_azulada():
    img = cm.make_palette_image()
    rng = cm.HSVRange(h_min=110, h_max=125, s_min=150)
    mask = cm.hsv_mask(img, rng)
    assert cv2.countNonZero(mask) > 0
    r, g, b = cm.mask_mean_color(img, mask)
    assert b > r and b > g


def test_mascara_que_da_a_volta_no_vermelho():
    img = cm.make_palette_image()
    rng = cm.HSVRange(h_min=172, h_max=8, s_min=150)
    r, g, b = cm.mask_mean_color(img, cm.hsv_mask(img, rng))
    assert r > 0.8 and g < 0.4 and b < 0.4


def test_mascara_vazia_retorna_none():
    img = np.zeros((10, 10, 3), np.uint8)  # imagem preta: V=0 fica fora do v_min
    assert cm.mask_mean_color(img, cm.hsv_mask(img, cm.HSVRange())) is None


def test_hsvrange_ajustes_permanecem_validos():
    r = cm.HSVRange(h_min=170, h_max=10)
    r.shift_hue(15)
    assert (r.h_min, r.h_max) == (5, 25)
    r.widen(100)
    assert 0 <= r.h_min < 180 and 0 <= r.h_max < 180
    r.shift_saturation(-1000)
    assert r.s_min == 0


def test_painel_de_analise_tem_4_quadros():
    img = cm.make_palette_image(100, 50)
    mask = cm.hsv_mask(img, cm.HSVRange())
    panel = cm.analysis_preview(img, mask, cm.mask_mean_color(img, mask))
    assert panel.shape == (50, 400, 3)
