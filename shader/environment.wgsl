const ENV_WIDTH: f32 = 1536.0;
const ENV_HEIGHT: f32 = 384.0;
const ENV_SIZE: f32 = 512.0;
const CLOUD_SHADOW_EXTENT: f32 = 760.0;
const CLOUD_MAP_WIDTH: f32 = 194.0;
const CLOUD_MAP_HEIGHT: f32 = 96.0;
const CLOUD_MIN_ELEVATION: f32 = -0.08;
const CLOUD_ELEVATION_SPAN: f32 = 0.5 * PHOTON_PI - CLOUD_MIN_ELEVATION;

// High-resolution upper-hemisphere cloud map (2048x456).
// All vertical rows cover [-0.08 rad, +pi/2] so zero resolution is wasted
// below the horizon, with two periodic guard columns on each side for cubic filtering.
fn cloud_map_texel(direction: vec3<f32>) -> vec2<f32> {
    var longitude = 0.0;
    if any(direction.xz != vec2(0.0)) {
        longitude = atan2(direction.x, direction.z);
    }
    let u = fract(longitude / TAU + 0.5);
    let elev = asin(clamp(direction.y, -1.0, 1.0));
    let v = clamp((elev - CLOUD_MIN_ELEVATION) / CLOUD_ELEVATION_SPAN, 0.0, 1.0);
    return vec2(2.0 + u * (ENV_WIDTH - 4.0), 0.5 + v * (ENV_HEIGHT - 1.0));
}

fn cloud_map_direction(texel: vec2<f32>) -> vec3<f32> {
    let longitude = TAU * (fract((texel.x - 2.0) / (ENV_WIDTH - 4.0)) - 0.5);
    let v = (texel.y - 0.5) / (ENV_HEIGHT - 1.0);
    let elevation = CLOUD_MIN_ELEVATION + v * CLOUD_ELEVATION_SPAN;
    let horizontal = cos(elevation);
    return vec3(horizontal * sin(longitude), sin(elevation), horizontal * cos(longitude));
}
