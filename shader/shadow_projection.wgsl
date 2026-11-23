override shadows: bool = true;
override classic_shadow_map: bool = true;

// Unified multi-scale shadow cascade distances (single source of truth shared across
// material.wgsl, world_lighting.wgsl, contact_ao.wgsl, and heightfield_compute.wgsl):
//   Tier 0 (0..24m):     Bend Studio near contact shadows
//   Tier 1 (0..144m):    2048x2048 native 3D shadow map (full strength to 114m)
//   Tier 2/3 (96..512m): Linear-Z Hi-Z SSRT + Decima scrolled world-space heightfield cache
const SHADOW_MAP_DIST: f32 = 144.0;
const SHADOW_MAP_DIST_SQ: f32 = 20736.0;
const SHADOW_FADE_START: f32 = 114.0;
const SHADOW_FADE_END: f32 = 140.0;
const SHADOW_ASSIST_START: f32 = 96.0;

const RAY_SHADOW_SIZE: f32 = 512.0;
const RAY_SHADOW_DEPTH: f32 = 192.0;
const RAY_SHADOW_WATER_FLAG: u32 = 31u;

fn pack_shadow_active() -> bool {
    return shadows && frame.has_shadow != 0u;
}

fn ray_shadow_basis() -> mat3x3<f32> {
    let light = frame.light_direction;
    // Rotate the world X axis with the light from the zenith. This basis is
    // continuous throughout the upper hemisphere, including at noon.
    let right = vec3(1.0, 0.0, 0.0) - light.x * (light + vec3(0.0, 1.0, 0.0)) / (1.0 + light.y);
    return mat3x3<f32>(right, cross(light, right), light);
}

fn ray_shadow_grid() -> vec3<f32> {
    return vec3(vec2(2.0 * 96.0 / RAY_SHADOW_SIZE), 1.0);
}

fn ray_shadow_phase_pixel() -> vec2<i32> {
    // The cloud shadow strip leaves the rest of this row unused.
    return vec2(i32(CLOUD_MAP_WIDTH), i32(ENV_SIZE));
}

fn ray_shadow_atlas_pixel(cell: vec2<i32>) -> vec2<i32> {
    // Store the square shadow grid in contiguous rows below the cloud strip.
    let index = cell.y * i32(RAY_SHADOW_SIZE) + cell.x;
    return vec2(index % i32(ENV_SIZE), index / i32(ENV_SIZE) + i32(ENV_SIZE + CLOUD_MAP_HEIGHT));
}

fn pack_shadow_edge_fade(position: vec3<f32>, projected: vec3<f32>) -> f32 {
    let edge = abs(2.0 * projected - 1.0);
    let atlas = smoothstep(0.81, 1.0, max(edge.x, edge.y));
    if frame.has_shadow != 0u {
        // Keep full 3D shadow-map strength out to SHADOW_FADE_START (114m), then blend
        // seamlessly over 114..140m (circular) and 0.84..0.99 (clip) into Hi-Z + Heightfield.
        let clip = frame.shadow_view_proj * vec4(position, 1.0);
        let edge_xy = abs(clip.xy) / max(abs(clip.w), 1e-5);
        let clip_fade = smoothstep(0.84, 0.99, max(edge_xy.x, edge_xy.y));
        let radial_fade = smoothstep(SHADOW_FADE_START, SHADOW_FADE_END, length(position));
        return max(clip_fade, radial_fade);
    }
    let horizontal = max(abs(position.x), abs(position.z));
    let scene = smoothstep(0.77, 0.97, horizontal / 96.0);
    return max(max(atlas, scene), smoothstep(0.86, 1.0, edge.z));
}

fn pack_shadow_texel() -> f32 {
    return select(ray_shadow_grid().x, frame.shadow_center_texel, frame.has_shadow != 0u);
}

fn pack_shadow_depth_per_block() -> f32 {
    let projection = frame.shadow_view_proj;
    return select(0.5 / RAY_SHADOW_DEPTH, length(vec3(projection[0].z, projection[1].z, projection[2].z)),
        frame.has_shadow != 0u);
}
