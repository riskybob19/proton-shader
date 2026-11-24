const SKY_AZIMUTH_TEXELS: u32 = 128u;
const SKY_ELEVATION_TEXELS: u32 = 64u;
const SKY_LIGHT_ROW: i32 = i32(SKY_ELEVATION_TEXELS);
const TAU: f32 = 6.283185307;
const PHOTON_PI: f32 = TAU * 0.5;
const PHOTON_ISOTROPIC: f32 = 0.5 / TAU;

// Photon uses Rec.2020 for its lighting and grade. Horizon's frame is Rec.709.
const REC709_TO_REC2020 = mat3x3<f32>(
    vec3(0.627404, 0.069097, 0.016391),
    vec3(0.329283, 0.919540, 0.088013),
    vec3(0.043313, 0.011362, 0.895595),
);
const REC2020_TO_REC709 = mat3x3<f32>(
    vec3(1.660491, -0.124550, -0.018151),
    vec3(-0.587641, 1.132900, -0.100579),
    vec3(-0.072850, -0.008349, 1.118730),
);

fn photon_dampen(x: f32) -> f32 {
    let unit = clamp(x, 0.0, 1.0);
    return unit * (2.0 - unit);
}

// Smooth Reinhard-HDR and normalized-distance packing for rgba8unorm history resources
// (preserves HDR range up to ~124x while halving render-size resource budget).
fn pack_hdr8(rgb: vec3<f32>) -> vec3<f32> {
    let c = max(rgb, vec3(0.0));
    return c / (c + vec3(1.0));
}

fn unpack_hdr8(rgb: vec3<f32>) -> vec3<f32> {
    let c = clamp(rgb, vec3(0.0), vec3(0.992));
    return c / (vec3(1.0) - c);
}

fn pack_dist8(d: f32) -> f32 {
    return clamp(d / (d + 48.0), 0.004, 0.996);
}

fn unpack_dist8(a: f32) -> f32 {
    let c = clamp(a, 0.0, 0.996);
    return 48.0 * c / max(1.0 - c, 1e-3);
}

fn pack_u32_vec4(v: vec4<f32>) -> vec4<u32> {
    return vec4<u32>(pack4x8unorm(clamp(v, vec4(0.0), vec4(1.0))), 0u, 0u, 0u);
}

fn unpack_u32_vec4(raw: vec4<u32>) -> vec4<f32> {
    return unpack4x8unorm(raw.x);
}

// Camera-rotation-locked, translation-free world ray for sky & clouds:
// Taking the difference between two unprojected depths cancels 100% of bobView
// eye translation while keeping the exact same camera rotation as terrain.
fn sky_world_ray(ndc: vec2<f32>) -> vec3<f32> {
    let p_near = frame.projection_inverse * vec4(ndc, 0.5, 1.0);
    let p_far = frame.projection_inverse * vec4(ndc, 0.01, 1.0);
    let view_dir = p_far.xyz / p_far.w - p_near.xyz / p_near.w;
    return normalize((frame.view_inverse * vec4(view_dir, 0.0)).xyz);
}

fn photon_phase(cosine: f32, anisotropy: f32) -> f32 {
    let g2 = anisotropy * anisotropy;
    let denominator = max(1.0 + g2 - 2.0 * anisotropy * cosine, 1e-5);
    let inv_s = inverseSqrt(denominator);
    return PHOTON_ISOTROPIC * (1.0 - g2) * (inv_s * inv_s * inv_s);
}

struct PhotonDaylight {
    sunrise: f32,
    noon: f32,
    sunset: f32,
    midnight: f32,
}

fn photon_daylight() -> PhotonDaylight {
    let elevation = frame.sun_direction.y;
    let fade = select(1.7, 0.37 + 1.2 * max(0.0, -elevation), elevation < 0.18);
    let twilight = clamp(1.0 - fade * abs(elevation - 0.18), 0.0, 1.0);
    let transition = twilight * twilight;
    let rising = frame.day_time < 0.25 || frame.day_time > 0.75;
    return PhotonDaylight(
        select(0.0, transition, rising),
        select(0.0, 1.0 - transition, elevation >= 0.0),
        select(transition, 0.0, rising),
        select(1.0 - transition, 0.0, elevation >= 0.0),
    );
}

// Moon phase supplies Horizon's exposed eight-day cycle. Daytime gives
// continuous changes within each day, independent of the frame clock.
fn weather_cycle() -> f32 {
    return (f32(frame.moon_phase) + frame.day_time) * (TAU / 8.0);
}

const FOREST_FOG_CELL = vec2(4, SKY_LIGHT_ROW);
const CLOUD_WEATHER_CELL = vec2(3, SKY_LIGHT_ROW);
const EXPOSURE_CELL = vec2(5, SKY_LIGHT_ROW);

// Approximate pigment filtering along the longer diffuse paths within a canopy.
// High-albedo canopies (e.g. cherry blossoms, pale oak) also experience stronger
// internal multiple-scattering cavity darkening so bright petals keep rich depth.
fn canopy_albedo(albedo: vec3<f32>) -> vec3<f32> {
    let pigment = clamp(albedo, vec3(0.0), vec3(1.0));
    let y = dot(pigment, frame.luminance);
    let filtered = pigment * sqrt(pigment);
    let normalized = filtered * (y / max(dot(filtered, frame.luminance), 1e-8));
    let cavity_damp = mix(1.0, 0.80, smoothstep(0.28, 0.65, y));
    return mix(albedo, min(normalized, vec3(1.0)), 0.32) * cavity_damp;
}

fn weather_humidity() -> f32 {
    return clamp(0.18 + 0.53 * frame.downfall + 0.14 * sin(2.0 * weather_cycle() + 0.7)
        + 0.38 * frame.wetness, 0.0, 1.0);
}

fn weather_wind() -> f32 {
    return clamp(0.42 + 0.24 * sin(3.0 * weather_cycle() + 1.7) + 0.42 * frame.thunder, 0.0, 1.0);
}

fn storm_flash() -> f32 {
    if frame.thunder <= 0.4 {
        return 0.0;
    }
    let age = frame.time * 0.14;
    let seed = fract(sin(floor(age) * 78.233) * 43758.5453);
    let phase = fract(age) - (0.1 + 0.73 * seed);
    let pulse = smoothstep(-0.012, 0.0, phase) * (1.0 - smoothstep(0.0, 0.022, phase));
    let after = smoothstep(0.030, 0.039, phase) * (1.0 - smoothstep(0.039, 0.057, phase));
    return smoothstep(0.4, 0.95, frame.thunder) * (pulse + 0.32 * after) * 0.75;
}

// Deterministic 3D storm-cell direction per lightning strike so clouds and sky
// light up from inside a localized storm bank (RDR2 in-cloud sheet lightning).
fn storm_flash_dir() -> vec3<f32> {
    let strike = floor(frame.time * 0.14);
    let az = fract(sin(strike * 127.1 + 31.7) * 43758.5453) * TAU;
    let el = 0.18 + 0.38 * fract(sin(strike * 269.5 + 11.3) * 43758.5453);
    let h = sqrt(max(1.0 - el * el, 0.0));
    return vec3(cos(az) * h, el, sin(az) * h);
}

fn lunar_brightness() -> f32 {
    return 0.12 + 0.88 * (0.5 + 0.5 * cos(f32(frame.moon_phase) * TAU / 8.0));
}

override underwater_fog: bool = true;
override wave_strength: f32 = 0.5;
override water_caustics: bool = true;
override biome_water_tint: bool = true;
override ssilvb_gi: bool = true;
override gi_rsm_bounce: bool = true;
override blocklight_mode: u32 = 2u;

fn use_underwater_fog() -> bool {
    return underwater_fog && frame.dimension == 0 && frame.eye_in == 1u
        && frame.blindness <= 0.0 && frame.darkness <= 0.0;
}

override aurora_mode: u32 = 1u;

fn aurora_amount() -> f32 {
    if aurora_mode == 0u || (aurora_mode == 1u
        && (frame.temperature >= 0.15 || (frame.moon_phase % 5u) != 1u)) {
        return 0.0;
    }
    return smoothstep(0.05, 0.28, -frame.sun_direction.y) * (1.0 - frame.rain);
}

