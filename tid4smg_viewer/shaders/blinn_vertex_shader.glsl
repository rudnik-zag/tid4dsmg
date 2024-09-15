attribute vec3 vertex_position;
attribute vec3 vertex_normal;
varying vec3 base_normal;

void main() {
    gl_Position = gl_ModelViewProjectionMatrix * vec4(vertex_position, 1.0);
    base_normal = gl_NormalMatrix * normalize(vertex_normal);
}