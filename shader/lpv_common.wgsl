override colored_lpv: bool = true;

// 3D Scrolled Colored Light Propagation Volume (LPV) packed into 512x256 rgba16float:
//   64 x 32 x 64 voxels @ 2.0m/voxel = 128 x 64 x 128 blocks around snapped camera anchor.
//   8 x 4 tile grid of 64x64 XZ slices (slice y lives at tile (y & 7, y >> 3)).
const LPV_XZ: f32 = 64.0;
const LPV_Y: f32 = 32.0;
const LPV_CELL: f32 = 2.0;
const LPV_ATLAS_W: f32 = 512.0;
const LPV_ATLAS_H: f32 = 256.0;
const LPV_ATLAS_Y_OFFSET: i32 = 512;
const WORLD_VOLUMES_H: f32 = 768.0;

fn lpv_anchor(camera_pos: vec3<f32>) -> vec3<f32> {
    return floor(camera_pos * (1.0 / LPV_CELL));
}

fn lpv_voxel_to_pixel(v: vec3<i32>) -> vec2<i32> {
    let tx = (v.y & 7) * 64;
    let ty = (v.y >> 3) * 64;
    return vec2(tx + v.x, LPV_ATLAS_Y_OFFSET + ty + v.z);
}

fn lpv_pixel_to_voxel(pixel: vec2<i32>) -> vec3<i32> {
    let tx = pixel.x >> 6;
    let ty = pixel.y >> 6;
    let vy = ty * 8 + tx;
    return vec3(pixel.x & 63, vy, pixel.y & 63);
}

// Fast 2-tap trilinear lookup into a 2D-sliced 3D LPV atlas (zero inter-slice bleed).
fn lpv_sample_slice(volume: texture_2d<f32>, local_xz: vec2<f32>, slice_y: i32) -> vec3<f32> {
    let sy = clamp(slice_y, 0, 31);
    let tile = vec2<f32>(f32((sy & 7) * 64), f32(LPV_ATLAS_Y_OFFSET + (sy >> 3) * 64));
    let clamped_xz = clamp(local_xz, vec2(0.5), vec2(LPV_XZ - 0.5));
    let uv = (tile + clamped_xz) / vec2(LPV_ATLAS_W, WORLD_VOLUMES_H);
    return max(textureSampleLevel(volume, scattering_atlas_sampler, uv, 0.0).rgb, vec3(0.0));
}

fn lpv_sample_trilinear(volume: texture_2d<f32>, world_pos: vec3<f32>, anchor: vec3<f32>) -> vec3<f32> {
    let local = world_pos * (1.0 / LPV_CELL) - anchor + vec3(0.5 * LPV_XZ, 0.5 * LPV_Y, 0.5 * LPV_XZ);
    if any(local <= vec3(0.5)) || any(local >= vec3(LPV_XZ - 0.5, LPV_Y - 0.5, LPV_XZ - 0.5)) {
        return vec3(0.0);
    }
    let y0 = i32(floor(local.y - 0.5));
    let fy = clamp(fract(local.y - 0.5), 0.0, 1.0);
    let s0 = lpv_sample_slice(volume, local.xz, y0);
    let s1 = lpv_sample_slice(volume, local.xz, y0 + 1);
    let edge = min(
        min(min(local.x, LPV_XZ - local.x), min(local.z, LPV_XZ - local.z)),
        min(local.y, LPV_Y - local.y),
    );
    let fade = smoothstep(0.5, 10.0, edge);
    return mix(s0, s1, fy) * fade;
}
