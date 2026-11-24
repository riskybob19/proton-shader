const ENV_WIDTH: f32 = 768.0;
const ENV_HEIGHT: f32 = 768.0;
const ENV_SIZE: f32 = 512.0;
const CLOUD_SHADOW_EXTENT: f32 = 760.0;
const CLOUD_MAP_WIDTH: f32 = 194.0;
const CLOUD_MAP_HEIGHT: f32 = 96.0;
const CLOUD_MAP_ORIGIN: vec2<f32> = vec2(768.0, 0.0);
const CLOUD_MIN_ELEVATION: f32 = -0.08;
const CLOUD_ELEVATION_SPAN: f32 = 0.5 * PHOTON_PI - CLOUD_MIN_ELEVATION;
const CLOUD_MIN_Y: f32 = -0.08;

// Conflict-free Nubis3 3D NVDF + SDF + amortized solar light volume packed into
// unused texels of cloud_aux (1024x832):
//   [0..768, 0..768):     pole-free octahedral hemisphere temporal cloud aux
//   [768..962, 0..96):    camera-centered cloud shadow & coverage map
//   [768..800, 128..160): octahedral directional skylight atlas
//   (768, 200):           NVDF metadata (anchor_x, anchor_z, weather_key, valid)
//   (769, 200):           Panorama fast-refresh TTL counter (frames remaining after jump)
//   [0..1024, 768..832):  64x16x64 3D NVDF (16 horizontal 64x64 Y-slices)
//     .r = dimensional_profile, .g = detail_type, .b = sdf_dist, .a = voxel_sun_depth
const NVDF_ORIGIN = vec2<i32>(0, 768);
const NVDF_META_PIXEL = vec2<i32>(768, 200);
const NVDF_REFRESH_PIXEL = vec2<i32>(769, 200);
const NVDF_RES_XZ: f32 = 64.0;
const NVDF_RES_Y: f32 = 16.0;
const NVDF_CELL_XZ: f32 = 64.0;
const NVDF_BASE_Y: f32 = 183.0;
const NVDF_TOP_Y: f32 = 343.0;
const NVDF_CELL_Y: f32 = 10.0;

fn nvdf_snap_anchor(cam_xz: vec2<f32>) -> vec2<f32> {
    return floor(cam_xz * (1.0 / NVDF_CELL_XZ)) * NVDF_CELL_XZ;
}

fn nvdf_voxel_to_pixel(v: vec3<i32>) -> vec2<i32> {
    return NVDF_ORIGIN + vec2<i32>(v.y * 64 + v.x, v.z);
}

fn nvdf_pixel_to_voxel(pixel: vec2<i32>) -> vec3<i32> {
    let rel = pixel - NVDF_ORIGIN;
    return vec3<i32>(rel.x & 63, rel.x >> 6, rel.y);
}

fn nvdf_voxel_world_pos(v: vec3<i32>, anchor: vec2<f32>) -> vec3<f32> {
    return vec3(
        anchor.x + (f32(v.x) - 31.5) * NVDF_CELL_XZ,
        NVDF_BASE_Y + (f32(v.y) + 0.5) * NVDF_CELL_Y,
        anchor.y + (f32(v.z) - 31.5) * NVDF_CELL_XZ,
    );
}

fn nvdf_read_meta(aux_tex: texture_2d<f32>) -> vec4<f32> {
    return textureLoad(aux_tex, NVDF_META_PIXEL, 0);
}

fn nvdf_in_bounds(world_pos: vec3<f32>, anchor: vec2<f32>) -> bool {
    return abs(world_pos.x - anchor.x) < 1984.0
        && abs(world_pos.z - anchor.y) < 1984.0
        && world_pos.y >= NVDF_BASE_Y
        && world_pos.y <= NVDF_TOP_Y;
}

fn nvdf_sample_slice(aux_tex: texture_2d<f32>, local_xz: vec2<f32>, slice_y: i32) -> vec4<f32> {
    let sy = clamp(slice_y, 0, 15);
    let clamped_xz = clamp(local_xz, vec2(0.5), vec2(NVDF_RES_XZ - 0.5));
    let origin = vec2<f32>(f32(NVDF_ORIGIN.x + sy * 64), f32(NVDF_ORIGIN.y));
    let uv = (origin + clamped_xz) / vec2<f32>(textureDimensions(aux_tex));
    return textureSampleLevel(aux_tex, scattering_atlas_sampler, uv, 0.0);
}

fn nvdf_sample_trilinear(aux_tex: texture_2d<f32>, world_pos: vec3<f32>, anchor: vec2<f32>) -> vec4<f32> {
    let local_xz = (world_pos.xz - anchor) * (1.0 / NVDF_CELL_XZ) + vec2(32.0);
    let local_y = clamp((world_pos.y - NVDF_BASE_Y) * (1.0 / NVDF_CELL_Y), 0.5, NVDF_RES_Y - 0.5);
    let y0 = i32(floor(local_y - 0.5));
    let fy = clamp(local_y - 0.5 - f32(y0), 0.0, 1.0);
    let s0 = nvdf_sample_slice(aux_tex, local_xz, y0);
    let s1 = nvdf_sample_slice(aux_tex, local_xz, y0 + 1);
    return mix(s0, s1, fy);
}

fn cloud_shadow_from_aux(aux_tex: texture_2d<f32>, position: vec3<f32>) -> f32 {
    if frame.light_direction.y <= 0.05 {
        return 1.0;
    }
    if frame.camera_position.y + position.y > NVDF_BASE_Y - 8.0 {
        return -1.0;
    }
    let projected = position.xz - frame.light_direction.xz * (position.y / frame.light_direction.y);
    let uv = 0.5 + projected * (1.0 / CLOUD_SHADOW_EXTENT);
    if any(uv <= vec2(0.01)) || any(uv >= vec2(0.99)) {
        return -1.0;
    }
    let cell = vec2(0.5) + uv * vec2(CLOUD_MAP_WIDTH - 1.0, CLOUD_MAP_HEIGHT - 1.0)
        + CLOUD_MAP_ORIGIN;
    return textureSampleLevel(aux_tex, scattering_atlas_sampler,
        cell / vec2<f32>(textureDimensions(aux_tex)), 0.0).r;
}

// Pole-free, crease-free Lambert Azimuthal Equal-Area Disk hemisphere map (768x768):
// covers direction.y in [-0.08, +1.0] with zenith (0, 1, 0) at (0.5, 0.5).
// 100% rotationally isotropic in azimuth — zero zenith pole spokes AND zero 45-degree
// octahedral cardinal-direction V-chevron creases at N/S/E/W.
const CLOUD_DISK_RADIUS: f32 = 0.495;

fn cloud_map_texel(direction: vec3<f32>) -> vec2<f32> {
    let y_remapped = clamp((direction.y - CLOUD_MIN_Y) / (1.0 - CLOUD_MIN_Y), 0.0, 1.0);
    let rho = sqrt(max(1.0 - y_remapped, 0.0));
    let xz_len = length(direction.xz);
    let dir_xz = select(vec2(0.0, 0.0), direction.xz / xz_len, xz_len > 1e-6);
    let uv = clamp(vec2(0.5) + dir_xz * (rho * CLOUD_DISK_RADIUS), vec2(0.0), vec2(1.0));
    return vec2(0.5) + uv * vec2(ENV_WIDTH - 1.0, ENV_HEIGHT - 1.0);
}

fn cloud_map_direction(texel: vec2<f32>) -> vec3<f32> {
    let uv = clamp((texel - vec2(0.5)) / vec2(ENV_WIDTH - 1.0, ENV_HEIGHT - 1.0), vec2(0.0), vec2(1.0));
    let d = (uv - vec2(0.5)) * (1.0 / CLOUD_DISK_RADIUS);
    let r2 = dot(d, d);
    let y_remapped = clamp(1.0 - r2, 0.0, 1.0);
    let dy = clamp(y_remapped * (1.0 - CLOUD_MIN_Y) + CLOUD_MIN_Y, -1.0, 1.0);
    let d_len = length(d);
    let horiz = sqrt(max(1.0 - dy * dy, 0.0));
    let xz = select(vec2(0.0, 0.0), d * (horiz / d_len), d_len > 1e-6);
    return vec3(xz.x, dy, xz.y);
}
