#version 330

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

in vec4 vertex_color[];
out vec4 line_color;

uniform vec2 viewport_size;
uniform float line_width;

void emit_line_vertex(vec4 position, vec2 offset, vec4 color, float side) {
    gl_Position = position + vec4(offset * position.w * side, 0.0, 0.0);
    line_color = color;
    EmitVertex();
}

void main() {
    vec4 first = gl_in[0].gl_Position;
    vec4 second = gl_in[1].gl_Position;
    vec2 first_ndc = first.xy / max(first.w, 0.00001);
    vec2 second_ndc = second.xy / max(second.w, 0.00001);
    vec2 direction = second_ndc - first_ndc;
    float length_squared = dot(direction, direction);
    if (length_squared < 0.0000001) {
        return;
    }
    direction *= inversesqrt(length_squared);
    vec2 perpendicular = vec2(-direction.y, direction.x);
    vec2 offset = perpendicular * line_width / max(viewport_size, vec2(1.0));

    emit_line_vertex(first, offset, vertex_color[0],  1.0);
    emit_line_vertex(first, offset, vertex_color[0], -1.0);
    emit_line_vertex(second, offset, vertex_color[1],  1.0);
    emit_line_vertex(second, offset, vertex_color[1], -1.0);
    EndPrimitive();
}

