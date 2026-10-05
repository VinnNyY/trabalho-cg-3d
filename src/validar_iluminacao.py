"""
Validação da Parte 3: compara os pixels que o OpenGL desenhou com a equação de
iluminação de Phong calculada em NumPy (cg3d.lighting.phong_shade).

Para cada sólido e algumas orientações, projeta o centro de cada face visível
na tela, lê a cor do pixel com glReadPixels e compara com a cor prevista.

    python src/validar_iluminacao.py
"""
import sys

import glfw
import numpy as np

from cg3d import transforms as tf
from cg3d.app import App
from cg3d.quaternion import Quaternion

ORIENTACOES = [
    Quaternion.from_axis_angle((0.2, 1.0, 0.0), 0.5),
    Quaternion.from_axis_angle((1.0, 0.3, 0.0), 1.1),
    Quaternion.from_axis_angle((0.4, -1.0, 0.6), 2.3),
]
TOLERANCIA = 0.03  # 3% — arredondamento de 8 bits + interpolação entre vértices


def main() -> int:
    app = App(1000, 700)
    app.init_window(visible=False)
    pior = 0.0
    for mesh_index, mesh in enumerate(app.meshes):
        for q in ORIENTACOES:
            app.reset(verbose=False)
            app.state.mesh_index = mesh_index
            app.state.show_hud = False
            app.state.show_light_marker = False
            app.trackball.orientation = q
            app.render()
            img = app.capture()
            h, w = img.shape[:2]
            m, pv = app.model_matrix(), app.projection_matrix() @ app.view_matrix()
            for i, shade in enumerate(app.face_lighting()):
                centro = tf.transform_point(m, mesh.face_centroid(i))
                normal = tf.normal_matrix(m) @ mesh.face_normals[i]
                if np.dot(normal, np.subtract(app.eye(), centro)) <= 0.05:
                    continue  # face de costas para a câmera (não aparece)
                clip = pv @ np.append(centro, 1.0)
                ndc = clip[:3] / clip[3]
                x, y = int((ndc[0] + 1) / 2 * w), int((1 - (ndc[1] + 1) / 2) * h)
                pixel = img[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3).mean(axis=0)[::-1] / 255.0
                erro = float(np.abs(pixel - shade.color).max())
                pior = max(pior, erro)
                print(f"{mesh.name:9s} face {i}  N.L={shade.n_dot_l:+.2f}  "
                      f"OpenGL={np.round(pixel, 3)}  NumPy={np.round(shade.color, 3)}  erro={erro:.3f}")
    app.shutdown()
    ok = pior <= TOLERANCIA
    print(f"\nMaior erro: {pior:.3f}  →  {'OK: OpenGL e a equação de Phong concordam' if ok else 'FALHOU'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
