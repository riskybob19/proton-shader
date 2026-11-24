fn post(input: PostInput) -> vec4<f32> {
    if !use_volume_composite() {
        return vec4(1.0, 1.0, 1.0, -1.0);
    }
    let depth = input.depth;
    let position = fog_position(input.uv, depth);
    let ray_length = length(position);
    let distance = fog_distance(ray_length, depth);
    let ray = position / max(ray_length, 1e-6);
    var transmittance = fog_transmittance_at(ray, depth, distance);
    if depth > 0.0 && !use_underwater_fog() {
        let cloud_in_front = smoothstep(CLOUD_BASE - 8.0, CLOUD_BASE + 24.0, frame.camera_position.y)
            * smoothstep(0.02, -0.08, ray.y);
        if cloud_in_front > 0.001 {
            let end_dist = min(max(frame.fog_render.y, 16.0), max(frame.render_distance, 32.0));
            let cyl_ratio = clamp(length(position.xz) / end_dist, 0.0, 1.0);
            let rim_t = smoothstep(0.84, 1.0, cyl_ratio);
            let rim_fade = rim_t * rim_t * (3.0 - 2.0 * rim_t) * cloud_in_front;
            transmittance = mix(transmittance, vec3(1.0), rim_fade);
        }
    }
    let scatter_size = vec2<i32>(textureDimensions(fog_scattering));
    let cell = clamp(vec2<i32>(input.uv * vec2<f32>(scatter_size)), vec2(0), scatter_size - 1);
    let scatter_rgb = textureLoad(fog_scattering, cell, 0).rgb;
    var cloud_a = 1.0;
    if !(use_underwater_fog() && depth > 0.0) {
        // On open sky, fog_scattering.rgb is pure cloud radiance: zero radiance guarantees
        // zero cloud opacity, skipping the transmittance march in 1 fetch.
        if depth > 0.0 || max(scatter_rgb.r, max(scatter_rgb.g, scatter_rgb.b)) > 1e-5 {
            let pixel = floor(input.uv * frame.resolution * 0.5);
            cloud_a = cloud_transmittance_at(ray, depth, ray_length, pixel);
        }
    }
    return vec4(transmittance * cloud_a, 1.0);
}
