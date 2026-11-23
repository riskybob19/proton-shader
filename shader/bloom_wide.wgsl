override bloom_strength: f32 = 0.14;

fn post(input: PostInput) -> vec4<f32> {
    if bloom_strength <= 0.0 {
        return vec4(0.0);
    }

    // 5-tap Dual-Kawase blur (45% fewer texture fetches than 9-tap Gaussian, smoother wide halo).
    let step_uv = 2.5 / vec2<f32>(textureDimensions(bloom_near));
    var color = 4.0 * textureSampleLevel(bloom_near, linear_sampler, input.uv, 0.0).rgb;
    color += textureSampleLevel(bloom_near, linear_sampler, input.uv + vec2(-step_uv.x, -step_uv.y), 0.0).rgb;
    color += textureSampleLevel(bloom_near, linear_sampler, input.uv + vec2(step_uv.x, -step_uv.y), 0.0).rgb;
    color += textureSampleLevel(bloom_near, linear_sampler, input.uv + vec2(-step_uv.x, step_uv.y), 0.0).rgb;
    color += textureSampleLevel(bloom_near, linear_sampler, input.uv + vec2(step_uv.x, step_uv.y), 0.0).rgb;
    return vec4(color * 0.125, 1.0);
}
