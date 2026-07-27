#version 330

in vec3 in_position;
in vec4 in_color;
in float in_size;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float pixel_ratio;
uniform float fog_intensity;

out vec4 vertex_color;
out float fog_amount;

void main() {
    vec4 camera_position = view * model * vec4(in_position, 1.0);
    gl_Position = projection * camera_position;
    float perspective_scale = clamp(8.0 / max(1.0, -camera_position.z), 0.55, 1.8);
    gl_PointSize = clamp(in_size * pixel_ratio * perspective_scale, 1.0, 10.0);
    vertex_color = in_color;
    float depth = clamp((-camera_position.z - 3.0) / 11.0, 0.0, 1.0);
    fog_amount = depth * fog_intensity;
}
