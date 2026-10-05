"""
Aplicação interativa: janela GLFW + contexto OpenGL + laço de renderização.

Pipeline de coordenadas aplicado a cada quadro (Parte 2):

    Objeto ──M = T·R·S──▶ Mundo ──V = lookAt──▶ Câmera ──P = perspective──▶ Recorte

    * PROJECTION ← P            (glLoadMatrixf)
    * MODELVIEW  ← V            → posiciona a luz no Mundo
    * MODELVIEW  ← V · M        → desenha o objeto
com R = R_trackball(quatérnio) · R_z · R_y · R_x  (Partes 2 e 4).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import cv2
import glfw
import numpy as np
from OpenGL.GL import (
    GL_BACK, GL_COLOR_BUFFER_BIT, GL_CULL_FACE, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_FILL, GL_FLAT, GL_FRONT_AND_BACK, GL_LEQUAL, GL_LIGHTING, GL_LINE,
    GL_MODELVIEW, GL_MODELVIEW_MATRIX, GL_MULTISAMPLE, GL_PROJECTION,
    GL_PROJECTION_MATRIX, GL_RENDERER, GL_RGB, GL_UNSIGNED_BYTE, GL_VERSION,
    glClear, glClearColor, glColor3f, glCullFace, glDepthFunc, glDisable, glEnable,
    glGetFloatv, glGetString, glLoadIdentity, glLoadMatrixf, glMatrixMode,
    glPolygonMode, glReadPixels, glShadeModel, glViewport,
)

from . import color as colormod
from . import transforms as tf
from .geometry import Mesh, all_meshes
from .lighting import Light, Material, apply_material, place_light, setup_lighting
from .quaternion import Quaternion
from .renderer import draw_axes, draw_grid, draw_light_marker, draw_mesh, draw_normals
from .trackball import Trackball

CONTROLES = """
=============================  CONTROLES  =============================
 Mouse (botão esq.) arrastar   Trackball virtual (quatérnios)
 Roda do mouse                 Zoom (aproxima/afasta a câmera)
 1 / 2 / 3                     Pirâmide / Octaedro / Tetraedro
 Setas | PgUp PgDn             Translação X,Y | Z            (matriz T)
 X Y Z  (Shift inverte)        Rotação nos eixos principais  (R_x R_y R_z)
 + / -                         Escala uniforme               (matriz S)
 Q/A  W/S  E/D                 Canal R / G / B  ±            (modelo RGB)
 H                             Gira o matiz (+20° no espaço HSV)
 O                             Material vindo da máscara HSV do OpenCV
 [ ]   , .   ; '               Máscara: desloca matiz | largura | sat. mín.
 V                             Abre/fecha a janela de análise do OpenCV
 L                             Liga/desliga iluminação
 Shift + Setas                 Move a luz
 N  F  G  C                    Normais | Wireframe | Eixos/grade | Culling
 Espaço                        Rotação automática (quatérnio incremental)
 T                             Confere matrizes sintéticas x GLU
 P                             Salva screenshot (cena + painel OpenCV)
 R                             Reinicia tudo        |  Esc  sai
=======================================================================
"""


@dataclass
class SceneState:
    mesh_index: int = 0
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    euler_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # X, Y, Z
    scale: float = 1.0
    camera_distance: float = 6.0
    fovy: float = 45.0
    use_opencv_color: bool = False
    show_normals: bool = False
    wireframe: bool = False
    show_helpers: bool = True
    cull: bool = True
    auto_rotate: bool = False


class App:
    def __init__(self, width: int = 1100, height: int = 750, title: str = "CG 3D — Trackball com Quatérnios"):
        self.width, self.height = width, height
        self.title = title
        self.meshes: list[Mesh] = all_meshes()
        self.state = SceneState()
        self.trackball = Trackball(sensitivity=1.5)
        self.light = Light()
        self.material = Material()
        self.manual_color = list(self.material.color)

        # OpenCV
        self.cv_image = colormod.make_palette_image()
        self.hsv_range = colormod.HSVRange()
        self.cv_color = None
        self.cv_window_open = False
        self._update_opencv_color()

        self.window = None
        self._last_time = 0.0
        self._screenshot_count = 0

    # ------------------------------------------------------------------ #
    # Inicialização (Parte 1)
    # ------------------------------------------------------------------ #
    def init_window(self, visible: bool = True) -> None:
        if not glfw.init():
            raise RuntimeError("Falha ao inicializar o GLFW")
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        if not visible:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Falha ao criar a janela/contexto OpenGL")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        self._init_gl()
        print(f"OpenGL {glGetString(GL_VERSION).decode()} — {glGetString(GL_RENDERER).decode()}")
        print(CONTROLES)

    def _init_gl(self) -> None:
        glClearColor(0.07, 0.08, 0.11, 1.0)
        # Teste de profundidade: cada fragmento só é escrito se estiver mais
        # perto da câmera que o já armazenado no Z-buffer.
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glShadeModel(GL_FLAT)          # uma normal por face → faces facetadas
        glEnable(GL_MULTISAMPLE)
        setup_lighting(self.light)

    # ------------------------------------------------------------------ #
    # Matrizes (Parte 2)
    # ------------------------------------------------------------------ #
    def projection_matrix(self) -> np.ndarray:
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        aspect = fb_w / max(fb_h, 1)
        return tf.perspective(self.state.fovy, aspect, 0.1, 100.0)

    def view_matrix(self) -> np.ndarray:
        eye = (0.0, 0.6, self.state.camera_distance)
        return tf.look_at(eye, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    def rotation_matrix(self) -> np.ndarray:
        rx, ry, rz = self.state.euler_deg
        euler = tf.compose(tf.rotation_z(rz), tf.rotation_y(ry), tf.rotation_x(rx))
        return self.trackball.matrix() @ euler

    def model_matrix(self) -> np.ndarray:
        t = tf.translation(*self.state.translation)
        s = tf.scale(self.state.scale, self.state.scale, self.state.scale)
        return tf.model_matrix(t, self.rotation_matrix(), s)

    # ------------------------------------------------------------------ #
    # Cores (Parte 3)
    # ------------------------------------------------------------------ #
    def _update_opencv_color(self) -> None:
        self.cv_mask = colormod.hsv_mask(self.cv_image, self.hsv_range)
        self.cv_color = colormod.mask_mean_color(self.cv_image, self.cv_mask)

    def current_color(self):
        if self.state.use_opencv_color and self.cv_color is not None:
            return self.cv_color
        return tuple(self.manual_color)

    def _report_color(self) -> None:
        r, g, b = self.current_color()
        h, s, v = colormod.rgb_to_hsv(r, g, b)
        hc, sc, vc = colormod.rgb_to_hsv_opencv(r, g, b)
        hm, sm, vm = colormod.hsv_degrees_to_opencv(h, s, v)
        origem = "OpenCV/máscara" if self.state.use_opencv_color else "manual"
        print(
            f"[cor {origem}] RGB=({r:.2f}, {g:.2f}, {b:.2f})  "
            f"HSV=({h:6.1f}°, {s:.2f}, {v:.2f})  "
            f"| escala OpenCV: nossa=({hm:.0f}, {sm:.0f}, {vm:.0f}) cv2=({hc}, {sc}, {vc})"
        )

    # ------------------------------------------------------------------ #
    # Renderização
    # ------------------------------------------------------------------ #
    def render(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(tf.to_gl(self.projection_matrix()))

        view = self.view_matrix()
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(tf.to_gl(view))

        setup_lighting(self.light)
        if self.light.enabled:
            place_light(self.light)               # luz no Espaço do Mundo

        if self.state.show_helpers:
            draw_grid()
            draw_axes()
            if self.light.enabled:
                draw_light_marker(self.light.position)

        glLoadMatrixf(tf.to_gl(view @ self.model_matrix()))

        self.material.color = self.current_color()
        apply_material(self.material)

        if self.state.cull and not self.state.wireframe:
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
        else:
            glDisable(GL_CULL_FACE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self.state.wireframe else GL_FILL)

        mesh = self.meshes[self.state.mesh_index]
        if self.state.wireframe:
            # Arestas não têm área iluminável: desenha com a cor pura do material.
            glDisable(GL_LIGHTING)
            glColor3f(*self.material.color)
            draw_mesh(mesh)
            setup_lighting(self.light)
        else:
            draw_mesh(mesh)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        if self.state.show_normals:
            draw_normals(mesh)

    def _update_title(self) -> None:
        mesh = self.meshes[self.state.mesh_index]
        q = self.trackball.orientation
        r, g, b = self.current_color()
        h, s, v = colormod.rgb_to_hsv(r, g, b)
        luz = "luz ON" if self.light.enabled else "luz OFF"
        cor = f"OpenCV {self.hsv_range}" if self.state.use_opencv_color else "manual"
        glfw.set_window_title(
            self.window,
            f"{mesh.name} | q=({q.w:+.2f},{q.x:+.2f},{q.y:+.2f},{q.z:+.2f}) | "
            f"RGB({r:.2f},{g:.2f},{b:.2f}) HSV({h:.0f}°,{s:.2f},{v:.2f}) | {cor} | {luz}",
        )

    def update(self, dt: float) -> None:
        if self.state.auto_rotate and not self.trackball.dragging:
            # Composição de quatérnios: pequena rotação em torno de Y por quadro.
            step = Quaternion.from_axis_angle((0.0, 1.0, 0.0), math.radians(45.0) * dt)
            self.trackball.orientation = (step * self.trackball.orientation).normalized()

    def run(self) -> None:
        self.init_window()
        self._last_time = glfw.get_time()
        title_timer = 0.0
        while not glfw.window_should_close(self.window):
            now = glfw.get_time()
            dt, self._last_time = now - self._last_time, now
            self.update(dt)
            self.render()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            title_timer += dt
            if title_timer > 0.1:
                self._update_title()
                title_timer = 0.0
            if self.cv_window_open:
                self._show_opencv_window()
        self.shutdown()

    def shutdown(self) -> None:
        if self.cv_window_open:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        glfw.terminate()

    # ------------------------------------------------------------------ #
    # OpenCV: janela de análise
    # ------------------------------------------------------------------ #
    def opencv_panel(self) -> np.ndarray:
        return colormod.analysis_preview(self.cv_image, self.cv_mask, self.cv_color)

    def _show_opencv_window(self) -> None:
        try:
            cv2.imshow("Analise HSV (OpenCV)", self.opencv_panel())
            cv2.waitKey(1)
        except cv2.error:
            print("Esta instalação do OpenCV não tem suporte a janelas (headless). "
                  "Use 'P' para salvar o painel em arquivo.")
            self.cv_window_open = False

    # ------------------------------------------------------------------ #
    # Screenshots
    # ------------------------------------------------------------------ #
    def capture(self) -> np.ndarray:
        w, h = glfw.get_framebuffer_size(self.window)
        data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR)

    def save_screenshot(self, path: str | None = None) -> str:
        self._screenshot_count += 1
        path = path or f"screenshot_{self._screenshot_count:02d}.png"
        cv2.imwrite(path, self.capture())
        panel_path = path.replace(".png", "_opencv.png")
        cv2.imwrite(panel_path, self.opencv_panel())
        print(f"Salvo: {path} e {panel_path}")
        return path

    # ------------------------------------------------------------------ #
    # Verificação contra a GLU (fidelidade matemática)
    # ------------------------------------------------------------------ #
    def verify_against_glu(self) -> None:
        try:
            from OpenGL.GLU import gluLookAt, gluPerspective
        except Exception as exc:  # pragma: no cover - depende do sistema
            print(f"GLU indisponível: {exc}")
            return
        if not (bool(gluPerspective) and bool(gluLookAt)):
            print("GLU não encontrada neste sistema (libGLU); as matrizes sintéticas "
                  "continuam validadas pelos testes automatizados (pytest).")
            return
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        aspect = fb_w / max(fb_h, 1)
        eye = (0.0, 0.6, self.state.camera_distance)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.state.fovy, aspect, 0.1, 100.0)
        glu_p = np.array(glGetFloatv(GL_PROJECTION_MATRIX)).reshape(4, 4).T

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*eye, 0, 0, 0, 0, 1, 0)
        glu_v = np.array(glGetFloatv(GL_MODELVIEW_MATRIX)).reshape(4, 4).T

        dp = np.abs(glu_p - self.projection_matrix()).max()
        dv = np.abs(glu_v - self.view_matrix()).max()
        ok = "OK" if max(dp, dv) < 1e-4 else "DIVERGENTE"
        print(f"[verificação] perspective x gluPerspective: erro máx = {dp:.2e}")
        print(f"[verificação] look_at     x gluLookAt     : erro máx = {dv:.2e}  → {ok}")

    # ------------------------------------------------------------------ #
    # Callbacks de entrada
    # ------------------------------------------------------------------ #
    def _on_resize(self, _win, width: int, height: int) -> None:
        glViewport(0, 0, width, max(height, 1))

    def _on_mouse_button(self, win, button, action, _mods) -> None:
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        x, y = glfw.get_cursor_pos(win)
        w, h = glfw.get_window_size(win)
        if action == glfw.PRESS:
            self.trackball.begin(x, y, w, h)
        elif action == glfw.RELEASE:
            self.trackball.end()

    def _on_cursor(self, win, x: float, y: float) -> None:
        if self.trackball.dragging:
            w, h = glfw.get_window_size(win)
            self.trackball.drag(x, y, w, h)

    def _on_scroll(self, _win, _dx: float, dy: float) -> None:
        self.state.camera_distance = float(np.clip(self.state.camera_distance - dy * 0.4, 2.5, 25.0))

    def _on_key(self, win, key, _scancode, action, mods) -> None:  # noqa: C901
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        st = self.state
        shift = bool(mods & glfw.MOD_SHIFT)
        step = 0.1

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
            st.mesh_index = key - glfw.KEY_1
            print(f"Modelo: {self.meshes[st.mesh_index].name}")

        # --- Translação / luz -------------------------------------------
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_UP, glfw.KEY_DOWN):
            dx = {glfw.KEY_LEFT: -1, glfw.KEY_RIGHT: 1}.get(key, 0)
            dy = {glfw.KEY_DOWN: -1, glfw.KEY_UP: 1}.get(key, 0)
            if shift:
                self.light.position[0] += dx * 0.3
                self.light.position[1] += dy * 0.3
            else:
                st.translation[0] += dx * step
                st.translation[1] += dy * step
        elif key == glfw.KEY_PAGE_UP:
            st.translation[2] -= step
        elif key == glfw.KEY_PAGE_DOWN:
            st.translation[2] += step

        # --- Rotação por eixo (matrizes R_x, R_y, R_z) --------------------
        elif key in (glfw.KEY_X, glfw.KEY_Y, glfw.KEY_Z):
            axis = {glfw.KEY_X: 0, glfw.KEY_Y: 1, glfw.KEY_Z: 2}[key]
            st.euler_deg[axis] = (st.euler_deg[axis] + (-5.0 if shift else 5.0)) % 360.0

        # --- Escala --------------------------------------------------------
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            st.scale = min(st.scale * 1.1, 4.0)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            st.scale = max(st.scale / 1.1, 0.2)

        # --- Cor RGB / HSV -------------------------------------------------
        elif key in (glfw.KEY_Q, glfw.KEY_A, glfw.KEY_W, glfw.KEY_S, glfw.KEY_E, glfw.KEY_D):
            channel = {glfw.KEY_Q: 0, glfw.KEY_A: 0, glfw.KEY_W: 1, glfw.KEY_S: 1,
                       glfw.KEY_E: 2, glfw.KEY_D: 2}[key]
            delta = 0.05 if key in (glfw.KEY_Q, glfw.KEY_W, glfw.KEY_E) else -0.05
            self.manual_color[channel] = float(np.clip(self.manual_color[channel] + delta, 0.0, 1.0))
            st.use_opencv_color = False
            self._report_color()
        elif key == glfw.KEY_H:
            h, s, v = colormod.rgb_to_hsv(*self.manual_color)
            if s < 0.05:  # cinza não tem matiz definido: dá saturação para ver a mudança
                s = 0.8
            self.manual_color = list(colormod.hsv_to_rgb(h + 20.0, s, max(v, 0.3)))
            st.use_opencv_color = False
            self._report_color()

        # --- OpenCV ---------------------------------------------------------
        elif key == glfw.KEY_O:
            st.use_opencv_color = not st.use_opencv_color
            if st.use_opencv_color and self.cv_color is None:
                print("Máscara vazia: ajuste os limites com [ ] , . ; '")
            self._report_color()
        elif key in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET, glfw.KEY_COMMA,
                     glfw.KEY_PERIOD, glfw.KEY_SEMICOLON, glfw.KEY_APOSTROPHE):
            if key == glfw.KEY_LEFT_BRACKET:
                self.hsv_range.shift_hue(-5)
            elif key == glfw.KEY_RIGHT_BRACKET:
                self.hsv_range.shift_hue(5)
            elif key == glfw.KEY_COMMA:
                self.hsv_range.widen(-3)
            elif key == glfw.KEY_PERIOD:
                self.hsv_range.widen(3)
            elif key == glfw.KEY_SEMICOLON:
                self.hsv_range.shift_saturation(-15)
            else:
                self.hsv_range.shift_saturation(15)
            self._update_opencv_color()
            st.use_opencv_color = True
            print(f"Máscara HSV: {self.hsv_range}")
            self._report_color()
        elif key == glfw.KEY_V:
            self.cv_window_open = not self.cv_window_open
            if not self.cv_window_open:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

        # --- Iluminação e visualização -------------------------------------
        elif key == glfw.KEY_L:
            self.light.enabled = not self.light.enabled
        elif key == glfw.KEY_N:
            st.show_normals = not st.show_normals
        elif key == glfw.KEY_F:
            st.wireframe = not st.wireframe
        elif key == glfw.KEY_G:
            st.show_helpers = not st.show_helpers
        elif key == glfw.KEY_C:
            st.cull = not st.cull
        elif key == glfw.KEY_SPACE:
            st.auto_rotate = not st.auto_rotate
        elif key == glfw.KEY_T:
            self.verify_against_glu()
        elif key == glfw.KEY_P:
            self.save_screenshot()
        elif key == glfw.KEY_R:
            self.reset()

    def reset(self) -> None:
        mesh_index = self.state.mesh_index
        self.state = SceneState(mesh_index=mesh_index)
        self.trackball.reset()
        self.light = Light()
        self.manual_color = list(Material().color)
        self.hsv_range = colormod.HSVRange()
        self._update_opencv_color()
        print("Cena reiniciada.")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    app = App()
    if argv and argv[0] in ("--imagem", "--image") and len(argv) > 1:
        img = cv2.imread(argv[1])
        if img is None:
            print(f"Não foi possível ler {argv[1]}; usando a paleta sintética.")
        else:
            app.cv_image = cv2.resize(img, (360, int(360 * img.shape[0] / img.shape[1])))
            app._update_opencv_color()
    app.run()
    return 0
