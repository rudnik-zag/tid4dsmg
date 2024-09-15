vec2 phong_weightCalc(in vec3 light_pos, in vec3 half_light, 
                      in vec3 frag_normal, in float shininess) {
    float n_dot_pos = max(0.0, dot(frag_normal, light_pos));
    float n_dot_half = 0.0;
    if (n_dot_pos > -.05) {
        n_dot_half = pow(max(0.0,dot(half_light, frag_normal)), shininess);
    }
    return vec2( n_dot_pos, n_dot_half);
}

uniform vec4 global_ambient;
uniform vec4 light_ambient;
uniform vec4 light_diffuse;
uniform vec4 light_specular;
uniform vec3 light_location;
uniform float material_shininess;
uniform vec4 material_specular;
uniform vec4 material_ambient;
uniform vec4 material_diffuse;
varying vec3 base_normal;

void main() {
    // normalized eye-coordinate light location
    // vec3 ec_light_location = normalize(
    //     gl_NormalMatrix * light_location
    // );
    vec3 ec_light_location = normalize(light_location);
    
    // half-vector calculation
    vec3 light_half = normalize(ec_light_location - vec3(0, 0, -1));
    vec2 weights = phong_weightCalc(
        ec_light_location,
        light_half,
        base_normal,
        material_shininess
    );
    
    gl_FragColor = clamp(
    (
        (global_ambient * material_ambient)
        + (light_ambient * material_ambient)
        + (light_diffuse * material_diffuse * weights.x)
        + (light_specular * material_specular * weights.y)
    ), 0.0, 1.0);
}