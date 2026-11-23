// Single-pass 2-tier Linear-Z Hierarchical Depth Pyramid (scale = 0.25):
// Stores linear view-space distance (meters) in rgba16float instead of raw Reversed-Z,
// preventing f16 subnormal underflow at 200..512 blocks:
//   .r = 4x4 min linear-Z (closest occluder for medium-range shadow rays, 0 = sky)
//   .g = 12x12 dilated min linear-Z (closest occluder for long-range mountain shadow rays)
//   .b = 4x4 average linear-Z (for macro valley/ridge AO)
fn hiz_linear_z(depth: f32) -> f32 {
    let p = frame.projection_inverse;
    let z = p[2].z * depth + p[3].z;
    let w = p[2].w * depth + p[3].w;
    return abs(z) / max(abs(w), 1e-6);
}

fn post(input: PostInput) -> vec4<f32> {
    if frame.dimension != 0 || (!shadows && contact_ao <= 0.0) {
        return vec4(0.0);
    }
    let full_size = vec2<i32>(textureDimensions(scene_depth));
    let hiz_size = vec2<i32>(max(vec2(1.0), floor(vec2<f32>(full_size) * 0.25)));
    let cell = clamp(vec2<i32>(input.uv * vec2<f32>(hiz_size)), vec2(0), hiz_size - 1);
    let base = cell * 4;
    var min_z4 = 65000.0;
    var sum_z = 0.0;
    var count = 0.0;
    for (var y = 0; y < 4; y += 1) {
        for (var x = 0; x < 4; x += 1) {
            let p = clamp(base + vec2(x, y), vec2(0), full_size - 1);
            let d = textureLoad(scene_depth, p, 0);
            if d > 0.0 {
                let lz = hiz_linear_z(d);
                min_z4 = min(min_z4, lz);
                sum_z += lz;
                count += 1.0;
            }
        }
    }
    var min_z12 = min_z4;
    let center = base + vec2(2, 2);
    let d0 = textureLoad(scene_depth, clamp(center + vec2(4, 0), vec2(0), full_size - 1), 0);
    if d0 > 0.0 { min_z12 = min(min_z12, hiz_linear_z(d0)); }
    let d1 = textureLoad(scene_depth, clamp(center + vec2(-4, 0), vec2(0), full_size - 1), 0);
    if d1 > 0.0 { min_z12 = min(min_z12, hiz_linear_z(d1)); }
    let d2 = textureLoad(scene_depth, clamp(center + vec2(0, 4), vec2(0), full_size - 1), 0);
    if d2 > 0.0 { min_z12 = min(min_z12, hiz_linear_z(d2)); }
    let d3 = textureLoad(scene_depth, clamp(center + vec2(0, -4), vec2(0), full_size - 1), 0);
    if d3 > 0.0 { min_z12 = min(min_z12, hiz_linear_z(d3)); }

    let out_z4 = select(0.0, min_z4, min_z4 < 60000.0);
    let out_z12 = select(0.0, min_z12, min_z12 < 60000.0);
    let avg_z = select(0.0, sum_z / count, count > 0.0);
    return vec4(out_z4, out_z12, avg_z, count * (1.0 / 16.0));
}
