#version 330

in vec4 vertex_color;
in float fog_amount;

uniform float glow_intensity;
uniform float white_core_intensity;
uniform vec3 background_color;

out vec4 frag_color;

void main() {
    vec2 delta = gl_PointCoord - vec2(0.5);
    float radius = length(delta) * 2.0;
    if (radius > 1.0) {
        discard;
    }
    float activity = smoothstep(0.46, 0.90, vertex_color.a);
    float core = 1.0 - smoothstep(0.10, 0.48, radius);
    float halo = (1.0 - smoothstep(0.14, 1.0, radius)) *
                 mix(0.28, 0.72, activity) * glow_intensity;
    float intensity = core + halo;
    vec3 color = mix(vertex_color.rgb, background_color, fog_amount);
    float white_mix = clamp(white_core_intensity, 0.0, 1.0) *
                      mix(0.18, 1.0, activity);
    vec3 core_color = mix(color, vec3(1.0), white_mix);
    core_color *= 1.0 + max(0.0, white_core_intensity - 1.0) *
                  0.65 * activity;
    float alpha = vertex_color.a * intensity * (1.0 - fog_amount * 0.55);
    frag_color = vec4(core_color * core + color * halo * 0.82, alpha);
}
