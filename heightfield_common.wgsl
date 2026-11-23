override cached_distant_shadows: bool = true;

// Decima-inspired scrolled world-space terrain heightfield cache:
// 512x512 rgba16float @ 2m/texel = 1024x1024m (+/-512m around snapped camera anchor).
// Channels: .r = cached world Y, .g = confidence (0..1), .b = source (1=shadow, 0.5=depth), .a = age.
const HF_SIZE: f32 = 512.0;
const HF_CELL_SIZE: f32 = 2.0;
const HF_EXTENT: f32 = 1024.0;

fn hf_anchor(camera_xz: vec2<f32>) -> vec2<f32> {
    return floor(camera_xz * (1.0 / HF_CELL_SIZE));
}

fn hf_world_to_uv(world_xz: vec2<f32>, anchor: vec2<f32>) -> vec2<f32> {
    let cell = world_xz * (1.0 / HF_CELL_SIZE) - anchor + 0.5 * HF_SIZE;
    return cell * (1.0 / HF_SIZE);
}
