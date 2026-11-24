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
    return vec4(transmittance, 1.0);
}
