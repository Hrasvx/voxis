#version 330

in vec2 uv;
uniform sampler2D current_frame;
uniform sampler2D previous_frame;
uniform float persistence;
uniform float glow_intensity;
uniform vec2 texel;

out vec4 frag_color;

void main() {
    vec3 current = texture(current_frame, uv).rgb;
    vec3 near_blur = vec3(0.0);
    near_blur += texture(current_frame, uv + texel * vec2( 1.0,  0.0)).rgb;
    near_blur += texture(current_frame, uv + texel * vec2(-1.0,  0.0)).rgb;
    near_blur += texture(current_frame, uv + texel * vec2( 0.0,  1.0)).rgb;
    near_blur += texture(current_frame, uv + texel * vec2( 0.0, -1.0)).rgb;
    near_blur *= 0.25;
    vec3 middle_blur = vec3(0.0);
    middle_blur += texture(current_frame, uv + texel * vec2( 3.0,  0.0)).rgb;
    middle_blur += texture(current_frame, uv + texel * vec2(-3.0,  0.0)).rgb;
    middle_blur += texture(current_frame, uv + texel * vec2( 0.0,  3.0)).rgb;
    middle_blur += texture(current_frame, uv + texel * vec2( 0.0, -3.0)).rgb;
    middle_blur += texture(current_frame, uv + texel * vec2( 2.0,  2.0)).rgb;
    middle_blur += texture(current_frame, uv + texel * vec2(-2.0,  2.0)).rgb;
    middle_blur += texture(current_frame, uv + texel * vec2( 2.0, -2.0)).rgb;
    middle_blur += texture(current_frame, uv + texel * vec2(-2.0, -2.0)).rgb;
    middle_blur *= 0.125;
    vec3 far_blur = vec3(0.0);
    far_blur += texture(current_frame, uv + texel * vec2( 7.0,  0.0)).rgb;
    far_blur += texture(current_frame, uv + texel * vec2(-7.0,  0.0)).rgb;
    far_blur += texture(current_frame, uv + texel * vec2( 0.0,  7.0)).rgb;
    far_blur += texture(current_frame, uv + texel * vec2( 0.0, -7.0)).rgb;
    far_blur *= 0.25;
    vec3 previous = texture(previous_frame, uv).rgb * persistence;
    vec3 bloom = near_blur * 0.10 + middle_blur * 0.055 + far_blur * 0.022;
    vec3 combined = max(current, previous) + bloom * glow_intensity;
    frag_color = vec4(combined, 1.0);
}
