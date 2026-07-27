#version 330

in vec2 uv;
uniform sampler2D image;
uniform float time;
uniform float grain_intensity;
uniform float flicker_intensity;
uniform float scanline_intensity;
uniform float chromatic_aberration;
uniform vec3 background_color;
uniform vec2 resolution;

out vec4 frag_color;

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    float aberration = 0.0008 * chromatic_aberration;
    float red = texture(image, uv + vec2(aberration, 0.0)).r;
    float green = texture(image, uv).g;
    float blue = texture(image, uv - vec2(aberration, 0.0)).b;
    vec3 color = vec3(red, green, blue);

    float noise = hash(uv * resolution + fract(time) * 173.0) - 0.5;
    float scanline = sin(uv.y * resolution.y * 1.55) *
                     0.035 * scanline_intensity;
    float flicker = 1.0 - flicker_intensity * 0.5 +
                    flicker_intensity * 0.5 * sin(time * 7.1);
    vec2 centered = uv * 2.0 - 1.0;
    float vignette = smoothstep(1.35, 0.32, dot(centered, centered));

    color = max(color, background_color);
    color *= flicker * mix(0.70, 1.0, vignette);
    color += (noise * grain_intensity + scanline * grain_intensity) *
             (0.35 + dot(color, vec3(0.333)));
    frag_color = vec4(max(color, vec3(0.0)), 1.0);
}
