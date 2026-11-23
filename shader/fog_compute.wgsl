fn compute(input: ComputeInput) {
    let size = textureDimensions(reflection_history);
    if !input.in_bounds || any(input.cell.xy >= size) {
        return;
    }
    if !terrain_reflections || frame.dimension != 0 {
        return;
    }
    let cell = vec2<i32>(input.cell.xy);
    let uv = (vec2<f32>(cell) + 0.5) / vec2<f32>(size);
    let depth = depth_at(uv);
    let position = fog_position(uv, depth);
    let ray_length = length(position);
    save_reflection_history(cell, uv, depth, position, min(ray_length, 2500.0));
}
