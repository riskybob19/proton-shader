override directional_skylight: bool = true;

// Packed into the unused right strip of cloud_aux (cols 768..799, rows 128..159),
// freeing a full persistent [[resource]] slot with zero VRAM overhead.
const SKYLIGHT_ATLAS_OFFSET: vec2<f32> = vec2(768.0, 128.0);
const SKYLIGHT_ATLAS_SIZE: f32 = 32.0;

fn octahedral_fold(plane: vec2<f32>) -> vec2<f32> {
    let signs = select(vec2(-1.0), vec2(1.0), plane >= vec2(0.0));
    return (1.0 - abs(plane.yx)) * signs;
}

fn skylight_atlas_uv(oct_uv: vec2<f32>, aux_size: vec2<f32>) -> vec2<f32> {
    let cell = SKYLIGHT_ATLAS_OFFSET + vec2(0.5) + oct_uv * (SKYLIGHT_ATLAS_SIZE - 1.0);
    return cell / aux_size;
}
