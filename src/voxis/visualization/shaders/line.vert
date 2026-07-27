#version 330

in vec3 in_position;
in vec4 in_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float fog_intensity;

out vec4 vertex_color;

void main() {
    vec4 camera_position = view * model * vec4(in_position, 1.0);
    gl_Position = projection * camera_position;
    float depth = clamp((-camera_position.z - 3.0) / 11.0, 0.0, 1.0);
    vertex_color = vec4(
        in_color.rgb,
        in_color.a * (1.0 - depth * fog_intensity * 0.75)
    );
}

