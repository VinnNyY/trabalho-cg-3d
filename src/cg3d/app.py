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
    GL_DEPTH_TEST, GL_FILL, GL_FRONT_AND_BACK, GL_LEQUAL, GL_LIGHTING,
    GL_LINE, GL_MODELVIEW, GL_MODELVIEW_MATRIX, GL_MULTISAMPLE,
    GL_PROJECTION, GL_PROJECTION_MATRIX, GL_RENDERER, GL_RGB,
    GL_UNSIGNED_BYTE, GL_VERSION, glClear, glClearColor, glColor3f,
    glCullFace, glDepthFunc, glDisable, glEnable, glGetFloatv, glGetString,
    glLoadIdentity, glLoadMatrixf, glMatrixMode, glPolygonMode, glReadPixels,
    glViewport,
)

from . import color as colormod
from . import transforms as tf
from .geometry import Mesh, all_meshes
from .hud import Hud
from .lighting import Light, Material, apply_material, phong_shade, place_light, setup_lighting
from .quaternion import Quaternion
from .renderer import draw_axes, draw_light, draw_mesh, draw_mesh_detailed, draw_normals
from .trackball import Trackball

CONTROLES = """
=============================  CONTROLES  =============================
 Mouse (botão esq.) arrastar   Trackball virtual (quatérnios)
 Roda do mouse                 Zoom (aproxima/afasta a câmera)
 1 / 2 / 3                     Pirâmide / Octaedro / Tetraedro
 Setas | PgUp PgDn             Translação X,Y | Z            (matriz T)
 X Y Z  (Shift inverte)        Rotação nos eixos principais  (R_x R_y R_z)
 + / -                         Escala uniforme               (matriz S)
 ---------------------------  ILUMINAÇÃO  ----------------------------
 L                             Liga/desliga toda a iluminação
 4 / 5 / 6                     Liga/desliga AMBIENTE / DIFUSA / ESPECULAR
 Shift + 4 / 5 / 6             Altera a intensidade da componente
 J / K                         Diminui / aumenta o brilho especular (n)
 Shift + Setas | Shift + PgUp/PgDn   Move a luz em X,Y | Z
 M                             Luz orbitando o objeto (anima a posição)
 U                             Luz PONTUAL (w=1) / DIRECIONAL (w=0)
 7                             Cor da luz (branca, quente, fria, verde)
 9                             Mostra/oculta o marcador da luz
 B                             Sombreamento: Detalhado (GL_SMOOTH) / Flat
 -----------------------------  CORES  --------------------------------
 Q/A  W/S  E/D                 Canal R / G / B  ±            (modelo RGB)
 H                             Gira o matiz (+20° no espaço HSV)
 O                             Material vindo da máscara HSV do OpenCV
 [ ]   , .   ; '               Máscara: desloca matiz | largura | sat. mín.
 V                             Abre/fecha a janela de análise do OpenCV
 ----------------------------  OUTROS  --------------------------------
 N  F  G  C                    Normais | Wireframe | Eixos | Culling
 I                             Mostra/oculta os painéis na tela
 Espaço                        Rotação automática (quatérnio incremental)
 T                             Confere matrizes sintéticas x GLU
 P                             Salva screenshot (cena + painel OpenCV)
 R                             Reinicia tudo        |  Esc  sai
=======================================================================
"""

INTENSITY_STEPS = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0]


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
    show_axes: bool = False
    show_hud: bool = True
    cull: bool = True
    auto_rotate: bool = False
    orbit_light: bool = False
    detailed_shading: bool = True
    show_light_marker: bool = True


class App:
    def __init__(self, width: int = 1200, height: int = 780, title: str = "CG 3D — Trackball com Quatérnios"):
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

        self.hud_light = Hud(0.5)
        self.hud_faces = Hud(0.48)
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
        glEnable(GL_MULTISAMPLE)
        setup_lighting(self.light)

    # ------------------------------------------------------------------ #
    # Matrizes (Parte 2)
    # ------------------------------------------------------------------ #
    def projection_matrix(self) -> np.ndarray:
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        aspect = fb_w / max(fb_h, 1)
        return tf.perspective(self.state.fovy, aspect, 0.1, 100.0)

    def eye(self) -> tuple[float, float, float]:
        return (0.0, 0.6, self.state.camera_distance)

    def view_matrix(self) -> np.ndarray:
        return tf.look_at(self.eye(), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))

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
    # Iluminação (Parte 3): a equação de Phong avaliada por face, em NumPy
    # ------------------------------------------------------------------ #
    def face_lighting(self):
        """Para cada face: normal e centro no Espaço do Mundo e o resultado da
        equação de iluminação. Usa a matriz normal (M⁻¹)ᵀ."""
        mesh = self.meshes[self.state.mesh_index]
        m = self.model_matrix()
        nm = tf.normal_matrix(m)
        results = []
        for i in range(len(mesh.faces)):
            n_world = nm @ mesh.face_normals[i]
            n_world /= np.linalg.norm(n_world)
            c_world = tf.transform_point(m, mesh.face_centroid(i))
            results.append(phong_shade(n_world, c_world, self.eye(), self.light, self.material))
        return results

    # ------------------------------------------------------------------ #
    # Renderização
    # ------------------------------------------------------------------ #
    def render(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, fb_w, max(fb_h, 1))

        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(tf.to_gl(self.projection_matrix()))

        view = self.view_matrix()
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(tf.to_gl(view))

        self.material.color = self.current_color()
        setup_lighting(self.light)
        if self.light.enabled:
            place_light(self.light)               # luz no Espaço do Mundo
            if self.state.show_light_marker:
                draw_light(self.light.position, self.light.color, self.state.translation)
        if self.state.show_axes:
            draw_axes()

        glLoadMatrixf(tf.to_gl(view @ self.model_matrix()))
        apply_material(self.material)

        if self.state.cull and not self.state.wireframe:
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
        else:
            glDisable(GL_CULL_FACE)

        mesh = self.meshes[self.state.mesh_index]
        if self.state.wireframe:
            # Arestas não têm área iluminável: desenha com a cor pura do material.
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_LIGHTING)
            glColor3f(*self.material.color)
            draw_mesh(mesh)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            setup_lighting(self.light)
        elif not self.light.enabled:
            # Sem iluminação: cor chapada — o objeto vira uma silhueta sem volume.
            glColor3f(*self.material.color)
            draw_mesh(mesh)
        elif self.state.detailed_shading:
            draw_mesh_detailed(mesh)
        else:
            draw_mesh(mesh)

        faces = self.face_lighting() if self.light.enabled else None
        if self.state.show_normals:
            draw_normals(mesh, intensities=[f.n_dot_l for f in faces] if faces else None)

        if self.state.show_hud:
            self._draw_hud(fb_w, fb_h, faces)

    def _draw_hud(self, fb_w: int, fb_h: int, faces) -> None:
        lt, st, mat = self.light, self.state, self.material
        on = lambda b: "ON " if b else "off"  # noqa: E731
        title = (240, 240, 245)
        dim = (165, 170, 185)
        amb_c, dif_c, spe_c = (120, 200, 255), (255, 210, 110), (255, 255, 255)
        p = lt.position
        tipo = "DIRECIONAL (w=0)" if lt.directional else "PONTUAL (w=1)"
        r, g, b = mat.color
        h, s, v = colormod.rgb_to_hsv(r, g, b)
        lines = [
            ("ILUMINACAO - modelo de Phong (Parte 3)", title),
            (f"Luz {('LIGADA' if lt.enabled else 'DESLIGADA')}: {tipo}  pos=({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})  cor {lt.color_name}", dim),
            (f"[4] Ambiente  {on(lt.use_ambient)} intensidade {lt.ambient_intensity:.2f}", amb_c),
            (f"[5] Difusa    {on(lt.use_diffuse)} intensidade {lt.diffuse_intensity:.2f}", dif_c),
            (f"[6] Especular {on(lt.use_specular)} intensidade {lt.specular_intensity:.2f}  brilho n={mat.shininess:.0f} [J/K]", spe_c),
            (f"Material kd=RGB({r:.2f},{g:.2f},{b:.2f}) HSV({h:.0f},{s:.2f},{v:.2f})  ka=0.6kd  ks={mat.specular[0]:.1f}", dim),
            (f"[B] Sombreamento: {'DETALHADO (GL_SMOOTH, faces subdivididas)' if st.detailed_shading else 'FLAT (GL_FLAT, 1 cor por face)'}", dim),
            (f"[M] Luz orbitando: {on(st.orbit_light)}   [U] tipo   [7] cor   [L] liga/desliga", dim),
        ]
        self.hud_light.draw(lines, fb_w, fb_h, "top-left")

        if faces:
            mesh = self.meshes[st.mesh_index]
            rows = [(f"{mesh.name}: I = ambiente + difusa + especular", title),
                    ("face   N.L     amb   dif   esp  -> I", dim)]
            for i, f in enumerate(faces):
                lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]  # noqa: E731
                tag = "quad" if len(mesh.faces[i]) == 4 else "tri "
                color = dif_c if f.n_dot_l > 0 else (110, 115, 130)
                rows.append((
                    f"{i} {tag} {f.n_dot_l:+.2f}   {lum(f.ambient):.2f}  {lum(f.diffuse):.2f}  "
                    f"{lum(f.specular):.2f} -> {lum(f.color):.2f}", color))
            rows.append(("N.L < 0: face de costas para a luz (so ambiente)", dim))
            self.hud_faces.draw(rows, fb_w, fb_h, "bottom-right")

    def _update_title(self) -> None:
        mesh = self.meshes[self.state.mesh_index]
        q = self.trackball.orientation
        glfw.set_window_title(
            self.window,
            f"{mesh.name} | q=({q.w:+.2f},{q.x:+.2f},{q.y:+.2f},{q.z:+.2f}) | "
            f"{self.light.components_label()}",
        )

    def update(self, dt: float) -> None:
        if self.state.auto_rotate and not self.trackball.dragging:
            # Composição de quatérnios: pequena rotação em torno de Y por quadro.
            step = Quaternion.from_axis_angle((0.0, 1.0, 0.0), math.radians(45.0) * dt)
            self.trackball.orientation = (step * self.trackball.orientation).normalized()
        if self.state.orbit_light:
            # gira a posição da luz em torno do eixo Y do mundo (60°/s)
            rot = tf.rotation_y(60.0 * dt)
            p = self.light.position
            x, y, z = (rot @ np.array([p[0], p[1], p[2], 0.0]))[:3]
            self.light.position = [x, y, z, p[3]]

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

    def save_screenshot(self, path: str | None = None, panel: bool = True) -> str:
        self._screenshot_count += 1
        path = path or f"screenshot_{self._screenshot_count:02d}.png"
        cv2.imwrite(path, self.capture())
        if panel:
            cv2.imwrite(path.replace(".png", "_opencv.png"), self.opencv_panel())
        print(f"Salvo: {path}")
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
            print("GLU não encontrada neste sistema (sudo apt install libglu1-mesa); "
                  "as matrizes sintéticas continuam validadas pelo pytest.")
            return
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        aspect = fb_w / max(fb_h, 1)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.state.fovy, aspect, 0.1, 100.0)
        glu_p = np.array(glGetFloatv(GL_PROJECTION_MATRIX)).reshape(4, 4).T

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*self.eye(), 0, 0, 0, 0, 1, 0)
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

    def _cycle_intensity(self, attr: str) -> None:
        current = getattr(self.light, attr)
        idx = min(range(len(INTENSITY_STEPS)), key=lambda i: abs(INTENSITY_STEPS[i] - current))
        setattr(self.light, attr, INTENSITY_STEPS[(idx + 1) % len(INTENSITY_STEPS)])

    def _on_key(self, win, key, _scancode, action, mods) -> None:  # noqa: C901
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        st, lt = self.state, self.light
        shift = bool(mods & glfw.MOD_SHIFT)
        step = 0.1

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
            st.mesh_index = key - glfw.KEY_1
            print(f"Modelo: {self.meshes[st.mesh_index].name}")

        # --- Translação / posição da luz -----------------------------------
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_UP, glfw.KEY_DOWN):
            dx = {glfw.KEY_LEFT: -1, glfw.KEY_RIGHT: 1}.get(key, 0)
            dy = {glfw.KEY_DOWN: -1, glfw.KEY_UP: 1}.get(key, 0)
            if shift:
                lt.position[0] += dx * 0.3
                lt.position[1] += dy * 0.3
            else:
                st.translation[0] += dx * step
                st.translation[1] += dy * step
        elif key in (glfw.KEY_PAGE_UP, glfw.KEY_PAGE_DOWN):
            dz = -1 if key == glfw.KEY_PAGE_UP else 1
            if shift:
                lt.position[2] += dz * 0.3
            else:
                st.translation[2] += dz * step

        # --- Rotação por eixo (matrizes R_x, R_y, R_z) --------------------
        elif key in (glfw.KEY_X, glfw.KEY_Y, glfw.KEY_Z):
            axis = {glfw.KEY_X: 0, glfw.KEY_Y: 1, glfw.KEY_Z: 2}[key]
            st.euler_deg[axis] = (st.euler_deg[axis] + (-5.0 if shift else 5.0)) % 360.0

        # --- Escala --------------------------------------------------------
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            st.scale = min(st.scale * 1.1, 4.0)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            st.scale = max(st.scale / 1.1, 0.2)

        # --- Iluminação ----------------------------------------------------
        elif key == glfw.KEY_L:
            lt.enabled = not lt.enabled
        elif key in (glfw.KEY_4, glfw.KEY_5, glfw.KEY_6):
            comp = {glfw.KEY_4: "ambient", glfw.KEY_5: "diffuse", glfw.KEY_6: "specular"}[key]
            if shift:
                self._cycle_intensity(f"{comp}_intensity")
            else:
                setattr(lt, f"use_{comp}", not getattr(lt, f"use_{comp}"))
            print(f"[luz] {lt.components_label()}")
        elif key == glfw.KEY_J:
            self.material.shininess = max(self.material.shininess / 1.4, 1.0)
        elif key == glfw.KEY_K:
            self.material.shininess = min(self.material.shininess * 1.4, 128.0)
        elif key == glfw.KEY_M:
            st.orbit_light = not st.orbit_light
        elif key == glfw.KEY_U:
            lt.toggle_type()
        elif key == glfw.KEY_7:
            lt.next_color()
        elif key == glfw.KEY_B:
            st.detailed_shading = not st.detailed_shading
        elif key == glfw.KEY_9:
            st.show_light_marker = not st.show_light_marker

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

        # --- Visualização ---------------------------------------------------
        elif key == glfw.KEY_N:
            st.show_normals = not st.show_normals
        elif key == glfw.KEY_F:
            st.wireframe = not st.wireframe
        elif key == glfw.KEY_G:
            st.show_axes = not st.show_axes
        elif key == glfw.KEY_C:
            st.cull = not st.cull
        elif key == glfw.KEY_I:
            st.show_hud = not st.show_hud
        elif key == glfw.KEY_SPACE:
            st.auto_rotate = not st.auto_rotate
        elif key == glfw.KEY_T:
            self.verify_against_glu()
        elif key == glfw.KEY_P:
            self.save_screenshot()
        elif key == glfw.KEY_R:
            self.reset()

    def reset(self, verbose: bool = True) -> None:
        mesh_index = self.state.mesh_index
        self.state = SceneState(mesh_index=mesh_index)
        self.trackball.reset()
        self.light = Light()
        self.material = Material()
        self.manual_color = list(self.material.color)
        self.hsv_range = colormod.HSVRange()
        self._update_opencv_color()
        if verbose:
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
