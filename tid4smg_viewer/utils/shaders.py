import os

from glumpy import gloo
from pathlib import Path


def get_shaders_by_name(name, base_path=None):
    if base_path is None:
        base_path = Path(os.path.dirname(os.path.realpath(__file__)))

    with open(base_path / f"../shaders/{name}_vertex_shader.glsl") as f:
        vertex_shader = f.read()

    with open(base_path / f"../shaders/{name}_fragment_shader.glsl") as f:
        fragment_shader = f.read()

    return gloo.Program(vertex_shader, fragment_shader)
