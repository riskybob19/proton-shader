// Pass 2 of Voxel-Strict Rotated 5-Tap A-Trous GI Denoiser (diagonal, step = 2).
fn post(input: PostInput) -> vec4<f32> {
    return gi_atrous_filter(indirect_gi_a, input, 2, true);
}
