// Pass 1 of Voxel-Strict Rotated 5-Tap A-Trous GI Denoiser (cardinal, step = 1).
fn post(input: PostInput) -> vec4<f32> {
    return gi_atrous_filter(indirect_gi_raw, input, 2, true);
}
