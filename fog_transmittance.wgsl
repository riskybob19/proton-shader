fn post(input: PostInput) -> vec4<f32> {
    if !use_volume_composite() {
        return vec4(1.0, 1.0, 1.0, -1.0);
    }
    let depth = input.depth;
    let position = fog_position(input.uv, depth);
    let ray_length = length(position);
    let distance = fog_distance(ray_length, depth);
    let ray = position / max(ray_length, 1e-6);
    let transmittance = fog_transmittance_at(ray, depth, distance);
    let scatter_size = vec2<i32>(textureDimensions(fog_scattering));
    let cell = clamp(vec2<i32>(input.uv * vec2<f32>(scatter_size)), vec2(0), scatter_size - 1);
    let cloud_a = textureLoad(fog_scattering, cell, 0).a;
    let guide = select(distance, -1.0, depth <= 0.0);
    return vec4(transmittance * cloud_a, guide);
}
