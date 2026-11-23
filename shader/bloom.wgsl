override bloom_strength: f32 = 0.14;

fn post(input: PostInput) -> vec4<f32> {
    if bloom_strength <= 0.0 { return vec4(0.0); }
    let texel = 1.0 / vec2<f32>(textureDimensions(taa_current));
    // Karis-weighted 4-tap downsample prevents single-pixel pre-TAA specular fireflies
    // from flickering in bloom.
    var color = vec3(0.0);
    var weight_sum = 0.0;
    for (var tap = 0u; tap < 4u; tap += 1u) {
        let offset = 2.0 * vec2<f32>(f32(tap & 1u), f32(tap >> 1u)) - 1.0;
        let c = max(textureSampleLevel(taa_current, linear_sampler, input.uv + offset * texel, 0.0).rgb, vec3(0.0));
        let luma = dot(c, frame.luminance);
        let soft = smoothstep(0.62, 1.72, luma);
        let karis = 1.0 / (1.0 + luma * 0.5);
        color += c * (soft * karis);
        weight_sum += karis;
    }
    return vec4(color / max(weight_sum, 1e-4), 1.0);
}
