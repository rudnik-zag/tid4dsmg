uniform vec4 ucolor;
attribute vec3 position;
attribute vec4 color;

varying vec4 v_color;

void main()
{
    v_color = ucolor * color;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(position,1.0);
}
