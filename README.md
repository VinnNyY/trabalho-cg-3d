# Aplicação Interativa 3D — Computação Gráfica e Visão Computacional

Aplicação em **Python 3** que integra modelagem geométrica por malha poligonal, transformações em coordenadas homogêneas, **iluminação de Phong** (ambiente, difusa e especular), modelos de cor RGB/HSV com análise no **OpenCV** e um **Trackball virtual com quatérnios** para girar o objeto com o mouse.

**Tecnologias:** Python 3 · PyOpenGL · GLFW · OpenCV (`cv2`) · NumPy · pytest

![Painel de iluminação](docs/iluminacao_painel.png)

*No canto superior esquerdo ficam os parâmetros da luz. No inferior direito aparece, para cada face, o N·L e as parcelas ambiente, difusa e especular, calculados em tempo real. As normais são coloridas de acordo com quanto a face recebe de luz.*

---

## Como executar

### Ubuntu / Linux
```bash
sudo apt install python3-venv python3-pip libglu1-mesa   # uma vez só
cd trabalho-cg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```
> Wayland (padrão no Ubuntu 24.04): se a janela não abrir, use `PYGLFW_LIBRARY_VARIANT=x11 python src/main.py`. Se a janela do OpenCV (tecla `V`) reclamar do Qt, acrescente `QT_QPA_PLATFORM=xcb`.

### Windows
```powershell
cd trabalho-cg
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/main.py
```

### Outros comandos
```bash
python src/main.py --imagem foto.jpg   # usa uma foto sua na análise de cor HSV
pytest                                 # 127 testes da parte matemática
python src/validar_iluminacao.py       # compara os pixels do OpenGL com a equação de Phong
```

---

## Estrutura do repositório

```
trabalho-cg/
├── src/
│   ├── main.py                  # ponto de entrada
│   ├── validar_iluminacao.py    # prova que o OpenGL desenha exatamente a equação de Phong
│   └── cg3d/
│       ├── app.py               # janela GLFW, contexto OpenGL, laço e controles  (Partes 1–4)
│       ├── geometry.py          # vértices/faces, normais, subdivisão das faces    (Partes 1 e 3)
│       ├── renderer.py          # GL_TRIANGLES / GL_QUADS, normais, fonte de luz visível
│       ├── transforms.py        # T, S, Rx, Ry, Rz, lookAt, perspective, frustum   (Parte 2)
│       ├── lighting.py          # GL_LIGHTING/GL_LIGHT0 + equação de Phong em NumPy (Parte 3)
│       ├── color.py             # RGB↔HSV, máscara HSV e cor média com OpenCV     (Parte 3)
│       ├── hud.py               # painéis de texto na tela (desenhados com OpenCV)
│       ├── quaternion.py        # quatérnios unitários e conversão para matriz 4×4  (Parte 4)
│       └── trackball.py         # mapeamento 2D → hemisfério e rotação do mouse    (Parte 4)
├── tests/                       # pytest: valida toda a matemática
├── docs/                        # imagens geradas pela própria aplicação
├── requirements.txt
├── pytest.ini
└── .gitignore                   # ignora .venv/, __pycache__/ etc.
```

---

## Controles

| Entrada | Ação |
|---|---|
| **Arrastar com o botão esquerdo** | Trackball virtual (quatérnios) |
| Roda do mouse | Zoom da câmera |
| `1` `2` `3` | Pirâmide / Octaedro / Tetraedro |
| Setas · `PgUp` `PgDn` | Translação em X, Y · Z (matriz **T**) |
| `X` `Y` `Z` (Shift inverte) | Rotação nos eixos principais (**Rx Ry Rz**) |
| `+` `-` | Escala uniforme (matriz **S**) |
| **Iluminação** | |
| `L` | Liga/desliga toda a iluminação |
| `4` · `5` · `6` | Liga/desliga a componente **ambiente · difusa · especular** |
| `Shift` + `4`/`5`/`6` | Muda a intensidade da componente (0 → 0,15 → … → 1) |
| `J` / `K` | Diminui / aumenta o brilho especular *n* |
| `Shift` + Setas · `Shift` + `PgUp`/`PgDn` | Move a luz em X, Y · Z |
| `M` | Luz orbitando o objeto |
| `U` | Luz **pontual** (w = 1) / **direcional** (w = 0) |
| `7` | Cor da luz: branca, quente, fria, verde |
| `B` | Sombreamento **detalhado** (GL_SMOOTH) / **flat** (GL_FLAT) |
| `9` | Mostra/oculta o marcador da luz |
| **Cores** | |
| `Q`/`A` · `W`/`S` · `E`/`D` | Aumenta/diminui os canais **R · G · B** |
| `H` | Gira o matiz +20° no espaço **HSV** |
| `O` | Usa a cor da máscara HSV do OpenCV como material |
| `[` `]` · `,` `.` · `;` `'` | Máscara: desloca matiz · largura da faixa · saturação mínima |
| `V` | Abre/fecha a janela de análise do OpenCV |
| **Outros** | |
| `N` · `F` · `G` · `C` · `I` | Normais · Wireframe · Eixos · Culling · Painéis na tela |
| `Espaço` | Rotação automática (composição incremental de quatérnios) |
| `T` | Compara as matrizes sintéticas com `gluLookAt`/`gluPerspective` |
| `P` | Salva screenshot da cena e do painel do OpenCV |
| `R` · `Esc` | Reinicia · Sai |

---

## Onde cada requisito foi atendido

### Parte 1 — Estrutura, ambiente e modelagem base
| Requisito | Implementação |
|---|---|
| `src/`, `requirements.txt`, `.gitignore` ignorando `.venv/` | raiz do repositório |
| Contexto via GLFW + `GL_DEPTH_TEST` | `App.init_window()` e `App._init_gl()` em `app.py` |
| Sólido definido por listas de vértices e faces | `pyramid()`, `octahedron()`, `tetrahedron()` em `geometry.py` |
| Malha com `GL_TRIANGLES` / `GL_QUADS` | `draw_mesh()` em `renderer.py`. A pirâmide usa **os dois**: laterais em triângulos e base em quad |

Todas as faces estão em ordem **anti-horária vista de fora**. Os testes comprovam isso de três formas: as normais apontam para fora, a malha é fechada e `V − A + F = 2`.

### Parte 2 — Transformações e espaços de coordenadas
| Requisito | Implementação |
|---|---|
| Matrizes 4×4 `T`, `S`, `Rx`, `Ry`, `Rz` | `translation`, `scale`, `rotation_x/y/z` em `transforms.py` |
| Composição `M = T · R · S` | `model_matrix()` em `transforms.py` e `App.model_matrix()` |
| Objeto → Mundo (Modelo) e Mundo → Câmera (Visão) | `App.model_matrix()` e `look_at()` (matriz sintética) |
| Perspectiva | `perspective()` e `frustum()` (sintéticas) |

As matrizes são escritas em NumPy como no quadro (vetor coluna) e enviadas com `glLoadMatrixf(M.T)`, já que o OpenGL é coluna-maior.

**Por que a ordem importa:** o vértice é multiplicado à direita, então `S` age primeiro, depois `R` e por último `T`. Assim o objeto escala e gira em torno do próprio centro antes de ser levado à posição. Com `R · T`, ele giraria em torno da origem do mundo, como se estivesse em órbita. O teste `test_ordem_importa` mostra essa diferença com números.

### Parte 3 — Normais, iluminação e cor

![Componentes da iluminação](docs/componentes_iluminacao.png)

**Normais**

| Requisito | Implementação |
|---|---|
| `N = (V1 − V0) × (V2 − V0)` | `face_normal()` em `geometry.py` |
| Normalização `\|N\| = 1` | `face_normal()`, com teste em `test_normais_unitarias` |
| `glNormal3fv()` por face | `draw_mesh()` em `renderer.py` |

**Iluminação e sombreamento**

| Requisito | Implementação |
|---|---|
| `glEnable(GL_LIGHTING)` e `glEnable(GL_LIGHT0)` | `setup_lighting()` em `lighting.py` |
| Componente **ambiente** | `GL_AMBIENT` da luz × `GL_AMBIENT` do material (tecla `4`) |
| Componente **difusa** | `GL_DIFFUSE`, lei de Lambert `max(N·L, 0)` (tecla `5`) |
| Componente **especular** | `GL_SPECULAR` + `GL_SHININESS`, termo `max(N·H, 0)ⁿ` (tecla `6`, `J`/`K`) |
| Sombreamento conforme a **posição da luz** | `place_light()` depois da Matriz de Visão. A luz é móvel (`Shift`+setas, `M`) e fica visível como uma esfera ligada ao objeto por um raio tracejado |

A equação que o OpenGL calcula, também implementada em NumPy em `phong_shade()`:

```
I = A_global·kₐ  +  A_luz·kₐ  +  D_luz·k_d·max(N·L, 0)  +  S_luz·k_s·max(N·H, 0)ⁿ
                    ambiente      difusa (Lambert)          especular (Blinn-Phong)

L = direção ponto → luz      V = direção ponto → olho      H = normalize(L + V)
```

Detalhes da implementação:

* **Cada componente pode ser ligada e desligada** (`4`/`5`/`6`), e a imagem acima mostra o efeito isolado de cada uma. Só com ambiente, o objeto perde o volume. A difusa revela a forma. A especular cria o reflexo que depende do observador.
* **O painel na tela mostra a equação face por face**: N·L e as parcelas ambiente, difusa e especular. Quando N·L < 0, a face está de costas para a luz e recebe só a parcela ambiente.
* **Pontual × direcional** (`U`): com `w = 1` os raios saem de um ponto e L muda ao longo da face. Com `w = 0` os raios são paralelos, como os do Sol.
* **Sombreamento detalhado** (`B`): o OpenGL fixo calcula a luz só nos vértices, e com 3 vértices por face não sobra lugar para formar um reflexo especular. Por isso cada face é subdividida em triângulos menores, mantendo **a mesma normal da face**. Assim a luz é avaliada em vários pontos e aparece o degradê da luz pontual. No modo flat, cada face recebe uma única cor (`GL_FLAT`).
* `GL_NORMALIZE` fica ligado porque a escala `S` muda o comprimento das normais. As normais usadas no painel passam pela matriz normal `(M⁻¹)ᵀ`.
* **Validação:** `python src/validar_iluminacao.py` desenha os três sólidos em várias orientações, lê os pixels com `glReadPixels` e compara com `phong_shade()`. A diferença máxima medida foi de **0,5%**.

**Cores e OpenCV**

| Requisito | Implementação |
|---|---|
| Cor dinâmica em RGB e conversão para HSV | teclas `Q A W S E D H`; `rgb_to_hsv()` e `hsv_to_rgb()` em `color.py` |
| Máscara por limites de matiz e saturação → material | `hsv_mask()` (`cv2.inRange`) e `mask_mean_color()` (`cv2.mean` com máscara) |
| Cor da luz × cor do material | tecla `7`: a luz quente/fria/verde multiplica a cor do objeto |

![Análise HSV](docs/analise_hsv_opencv.png)

| Cor vinda da máscara do OpenCV | Luz quente em material branco |
|---|---|
| ![Octaedro](docs/octaedro_opencv.png) | ![Tetraedro](docs/tetraedro_luz_quente.png) |

### Parte 4 — Quatérnios e Trackball
| Requisito | Implementação |
|---|---|
| Capturar clique e arraste | `_on_mouse_button()` e `_on_cursor()` em `app.py` |
| `(x, y)` da tela → `(x, y, z)` no hemisfério | `screen_to_ndc()` + `project_to_sphere()` em `trackball.py` |
| Eixo por produto vetorial, ângulo por produto escalar | `rotation_between()` em `trackball.py` |
| Quatérnio unitário `q = (w, x, y, z)` | `Quaternion.from_axis_angle()` em `quaternion.py` |
| Acumular `q_total = q_novo · q_total` | `Trackball.drag()`, com renormalização |
| Converter para 4×4 e enviar ao pipeline | `Quaternion.to_matrix()` → entra em `R` da matriz de modelo |

```
x² + y² ≤ 1  →  z = √(1 − x² − y²)
caso contrário →  (x, y) normalizado, z = 0   (na borda gira em torno do eixo de visão)
```

**Gimbal lock:** com ângulos de Euler, depois de girar 90° em Y, as rotações em X e Z passam a produzir o mesmo movimento e perde-se um grau de liberdade. O teste `test_sem_gimbal_lock` reproduz isso com as matrizes da Parte 2 e mostra que, com quatérnios, as duas rotações continuam diferentes.

---

## Testes (127)

* **Transformações:** rotações seguem a regra da mão direita. `RRᵀ = I`. `look_at` e `perspective` levam os pontos certos para os lugares certos. `perspective` coincide com `frustum`.
* **Malhas:** normais unitárias e apontando para fora. Faces planas. Malha fechada e orientada. A subdivisão preserva a área, a normal e a orientação.
* **Iluminação:**
  * a lei do cosseno vale (luz a 60° gera metade da difusa);
  * uma face de costas recebe só ambiente;
  * o especular é máximo quando H = N;
  * um *n* maior concentra o reflexo;
  * desligar uma componente zera aquela parcela;
  * a luz pontual e a direcional se comportam de forma diferente;
  * a cor da luz multiplica a cor do material.
* **Quatérnios:** `to_matrix()` dá o mesmo resultado que `Rx`/`Ry`/`Rz`. `i·j = k`. Compor quatérnios equivale a multiplicar matrizes. A norma continua 1 depois de 100 000 composições.
* **Trackball:** arrastar para a direita gira em +Y. Ir e voltar retorna à identidade.
* **Cores:** o RGB→HSV manual confere com `colorsys` e com `cv2.cvtColor`.

---

## Roteiro sugerido para a demonstração em sala

1. `python src/main.py`: aparece a pirâmide iluminada e a luz (esfera branca). Arraste com o mouse para usar o **trackball**.
2. **Iluminação:**
   * aperte `5` e `6` para ficar só com a ambiente (o objeto fica "chapado");
   * religue a difusa (`5`) para o volume aparecer;
   * religue a especular (`6`) para o reflexo aparecer;
   * aperte `N` e mostre no painel o **N·L** de cada face mudando enquanto você gira o objeto.
3. `M` faz a luz orbitar e as faces trocam de tom. `U` alterna para luz direcional. `7` muda a cor da luz. `J`/`K` mudam o brilho. `B` compara o modo detalhado com o flat.
4. Setas, `X`/`Y`/`Z` e `+`/`-` aplicam as matrizes **T, R, S**. `R` reinicia.
5. `Q`…`D` e `H` mudam a cor, e o terminal mostra a conversão **RGB → HSV** manual ao lado da do OpenCV.
6. `V` abre o painel do OpenCV. Com `[` e `]`, o objeto assume a **cor média da máscara**.
7. `2` e `3` trocam de sólido. `Espaço` liga a rotação automática por quatérnios.
8. Para fechar, rode `pytest` (127 testes passando) e `python src/validar_iluminacao.py`.
