import torch

from ctypes import c_void_p
from PIL import Image
from pathlib import Path

import numpy as np

import OpenGL.GL as gl
import OpenGL.arrays as gl_arrays


class PointcloudModel:
    def __init__(self, vertices: torch.Tensor):
        # setup  buffers
        self.vertices = vertices.detach().cpu().numpy()

        self.vertex_buffer = gl_arrays.vbo.VBO(self.vertices, usage=gl.GL_DYNAMIC_DRAW)

    def update(self, vertices: torch.Tensor):
        # only update the vertex buffer, needs the same number of vertices
        assert vertices.shape[0] == self.vertices.shape[0]
        self.vertices = vertices.detach().cpu().numpy()
        self.vertex_buffer.set_array(self.vertices)

    def draw(self, color: tuple[float] = (1.0, 1.0, 1.0, 1.0), point_size: float = 2.0):
        gl.glColor4f(*color)
        gl.glPointSize(point_size)

        self.vertex_buffer.bind()
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 12, c_void_p(0))

        gl.glDrawArrays(gl.GL_POINTS, 0, self.vertices.shape[0])
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        self.vertex_buffer.unbind()
