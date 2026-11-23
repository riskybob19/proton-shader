override bloom_strength: f32 = 0.14;

// 1/16-resolution wide Dual-Kawase atmospheric bloom halo (upstream Photon multi-level pyramid).
fn post(input: PostInput) -> vec4<f32> {
    if bloom_strength <= 0.0 {
        return vec4(0.0);
    }
    // 5-tap Dual-Kawase wide atmospheric halo (45% fewer fetches than 9-tap Gaussian).
    let step_uv = 5.5 / vec2<f32>(textureDimensions(bloom_wide));
    var color = 4.0 * textureSampleLevel(bloom_wide, linear_sampler, input.uv, 0.0).rgb;
    color += textureSampleLevel(bloom_wide, linear_sampler, input.uv + vec2(-step_uv.x, -step_uv.y), 0.0).rgb;
    color += textureSampleLevel(bloom_wide, linear_sampler, input.uv + vec2(step_uv.x, -step_uv.y), 0.0).rgb;
    color += textureSampleLevel(bloom_wide, linear_sampler, input.uv + vec2(-step_uv.x, step_uv.y), 0.0).rgb;
    color += textureSampleLevel(bloom_wide, linear_sampler, input.uv + vec2(step_uv.x, step_uv.y), 0.0).rgb;
    return vec4(color * 0.125, 1.0);
}
