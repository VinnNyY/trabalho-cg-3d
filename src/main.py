"""
Ponto de entrada.

    python src/main.py                     # paleta sintética para a análise HSV
    python src/main.py --imagem foto.jpg   # usa uma imagem sua na análise HSV
"""
import sys

from cg3d.app import main

if __name__ == "__main__":
    sys.exit(main())
