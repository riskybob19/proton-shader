override neural_atmosphere: bool = true;

// 16-wavelength (380-740nm) spectral spherical-shell multi-scattering atmosphere MLP (6 -> 16 -> 16 -> 6, ReLU).
// Rayleigh (lambda^-4) + Mie aerosols + Stratospheric Chappuis Ozone + Hillaire infinite-order MS -> CIE 1931 -> Rec.2020.
// Validation R2(log): bg=0.9995, mie=0.9982 | energy ratio vs 16-band spectral solver: bg=0.9988, mie=1.0054.
const NA_W1 = array<vec4<f32>, 24>(
    vec4(1.23872, -11.96885, 0.68765, 0.74943),
    vec4(0.50820, -2.61338, 1.21289, 0.59843),
    vec4(-0.43465, -0.00852, -0.17789, 0.14378),
    vec4(0.57904, 0.01471, 0.18283, 0.24711),
    vec4(0.12315, -0.00396, 0.09572, -0.50240),
    vec4(-0.12751, 0.00366, -0.08307, 0.54886),
    vec4(0.72312, -0.00007, 0.30820, 0.61444),
    vec4(1.58627, -1.53033, -0.18887, -0.38083),
    vec4(0.06610, -0.28879, -0.50217, -2.81334),
    vec4(-0.03583, -0.18100, 1.03313, 2.22367),
    vec4(0.25519, -0.21045, 0.09781, -0.03715),
    vec4(-0.26297, 0.19599, -0.05723, 0.02269),
    vec4(-0.03724, -0.18555, 0.60356, 1.55754),
    vec4(1.49731, -0.53962, 1.24632, 1.34898),
    vec4(0.21554, -0.85331, 0.47570, -0.22865),
    vec4(0.37411, -0.64958, -0.37358, 0.19513),
    vec4(0.42299, -0.09396, -0.33700, -0.05532),
    vec4(-0.41007, 0.12834, 0.39550, 0.05022),
    vec4(-0.89781, 0.76411, 0.02294, -0.10797),
    vec4(1.03287, -0.67027, 0.25138, 0.14716),
    vec4(0.78075, -3.52296, 0.82363, -0.14762),
    vec4(-0.86651, 1.00500, -0.18162, 0.64319),
    vec4(0.00852, -0.13790, 0.37124, -0.18343),
    vec4(0.00861, 0.10284, -0.41141, -0.33307),
);

const NA_B1 = array<vec4<f32>, 4>(
    vec4(0.14441, -0.07017, 0.04650, -0.03901),
    vec4(0.16102, 0.33934, -0.08054, 0.43964),
    vec4(-0.01790, 0.23130, 0.20086, 0.06513),
    vec4(0.21514, 0.27229, 0.16173, -0.02755),
);

const NA_W2 = array<vec4<f32>, 64>(
    vec4(-0.00229, -0.52555, -0.63656, 0.55366),
    vec4(-3.49379, -2.85511, -0.35533, -0.50307),
    vec4(-1.59817, 0.03970, -0.08082, -0.17128),
    vec4(-0.06277, 0.09237, 0.02689, 0.00937),
    vec4(0.87207, 0.20172, 0.34115, -0.08201),
    vec4(-0.32503, -1.47825, 0.76902, -0.10112),
    vec4(0.40374, 0.16990, 0.63436, -0.89502),
    vec4(-0.11295, -0.19446, -0.37217, -0.41501),
    vec4(-0.23192, 0.24427, -0.00449, -0.26508),
    vec4(0.02448, 0.69033, -0.22635, -0.72367),
    vec4(0.23059, 0.22682, 0.10878, 0.09468),
    vec4(-0.61137, -0.02795, -0.44271, -0.14539),
    vec4(-0.03951, 0.01688, 0.61687, 0.23476),
    vec4(-0.69161, -1.19046, -0.60676, 0.11448),
    vec4(0.09835, 0.03761, 0.05471, -0.44938),
    vec4(-0.06171, 0.06604, 0.02739, -0.20522),
    vec4(0.46645, 0.10780, -0.19944, -0.42124),
    vec4(-0.20206, -0.09420, -1.58517, -0.57916),
    vec4(-0.31082, -0.49667, -0.38453, 0.07156),
    vec4(0.21107, -0.15504, 0.57681, 0.26507),
    vec4(-0.09330, -0.09071, 0.10803, 0.08369),
    vec4(0.08605, -0.01896, -0.60627, -0.69271),
    vec4(-0.10522, -0.42963, -0.45202, -0.18151),
    vec4(-0.30506, -0.27950, -1.58260, -1.33979),
    vec4(-0.53167, 0.31177, 0.40580, 0.30568),
    vec4(0.55982, 0.36422, -0.14952, -0.52851),
    vec4(-0.29209, -0.42760, 0.67307, 0.15464),
    vec4(0.03516, -1.01005, -0.56080, 0.11378),
    vec4(0.03952, -0.39657, -0.05431, -0.05181),
    vec4(-0.27526, -0.30209, 0.96212, 0.31640),
    vec4(-0.25272, -0.44275, 0.89759, 0.21676),
    vec4(0.00022, -0.03381, 0.16359, 0.09587),
    vec4(-0.33618, -0.40077, 0.25630, -0.22238),
    vec4(-0.90745, -0.13934, 0.23860, -5.24967),
    vec4(-0.08000, -0.12315, -0.52288, -0.29589),
    vec4(0.06702, -0.31507, 0.04510, 0.06786),
    vec4(0.43554, 0.24854, 0.16227, 0.36711),
    vec4(0.14227, -0.92456, -0.65401, 0.68698),
    vec4(0.29387, 0.31245, -0.41792, 0.40419),
    vec4(-0.89669, -0.03057, -0.37751, 0.17648),
    vec4(-0.26697, -0.12945, -0.40066, 0.27807),
    vec4(-0.43149, -0.26090, 0.11862, 0.32585),
    vec4(0.25191, -0.28117, -0.32303, 0.75426),
    vec4(-0.06921, -0.63996, 0.25114, -0.81853),
    vec4(0.18378, -0.27931, -0.67492, 0.13174),
    vec4(-0.71741, 0.42871, 0.31166, -0.91098),
    vec4(0.43888, -0.35282, -0.10752, 0.12087),
    vec4(-0.00820, 0.36581, 0.17010, 0.14620),
    vec4(-0.52724, -0.82041, 0.67093, -0.48550),
    vec4(-0.66468, -0.03340, -7.17546, 0.08007),
    vec4(-1.04441, 0.99046, -1.58006, 0.22557),
    vec4(0.38416, -0.19771, -0.13642, -0.35134),
    vec4(0.72033, -0.24153, 0.90093, 0.04951),
    vec4(-0.35982, -0.69721, -0.09051, -0.24783),
    vec4(0.58922, 0.02025, -0.05122, 0.39630),
    vec4(0.05015, 0.35275, -0.10652, -0.33337),
    vec4(0.27769, -0.13482, 0.10718, -0.14748),
    vec4(0.06500, 0.49247, -0.04544, -0.17616),
    vec4(-0.00206, -0.07075, 0.60683, -0.23262),
    vec4(-0.54280, -0.39637, -0.60959, -0.27573),
    vec4(0.97876, 0.13845, -0.28083, 0.37740),
    vec4(-0.61015, -0.12793, -0.30183, 0.16139),
    vec4(-0.09006, -0.08864, -0.27580, -0.66126),
    vec4(0.07795, 0.14685, 0.03388, 0.11512),
);

const NA_B2 = array<vec4<f32>, 4>(
    vec4(0.29743, 0.33893, -0.18739, -0.22221),
    vec4(-0.16000, -0.06551, 0.12045, -0.49769),
    vec4(0.49598, -0.10827, -0.02134, -0.01955),
    vec4(0.24717, -0.11343, 0.07059, 0.00000),
);

const NA_W3_BG = array<vec3<f32>, 16>(
    vec3(0.18728, 0.02217, 0.02903),
    vec3(0.33947, 0.28061, 0.07292),
    vec3(0.56229, 0.26989, 0.01570),
    vec3(-0.04729, 0.48122, 0.47139),
    vec3(-0.10686, -0.02834, 0.15963),
    vec3(-0.31575, -0.35288, -0.21965),
    vec3(0.48120, 0.41815, 0.58266),
    vec3(-0.24335, -0.32293, -0.63218),
    vec3(-0.22266, 0.04660, 0.68130),
    vec3(-0.46865, -0.45765, 0.25144),
    vec3(0.12946, -0.60816, 0.84279),
    vec3(0.68913, 0.82445, 0.84674),
    vec3(0.55025, 0.63251, 0.60954),
    vec3(-1.02075, -0.34421, 0.03839),
    vec3(3.79571, 3.63180, 1.66354),
    vec3(-0.08012, -0.22363, 0.19641),
);

const NA_B3_BG = vec3(0.00139, -0.00044, 0.00426);

const NA_W3_MIE = array<vec3<f32>, 16>(
    vec3(1.43691, 1.27890, 0.88884),
    vec3(-0.27407, -0.17438, 0.11783),
    vec3(0.04170, 0.36200, 1.00838),
    vec3(-0.46621, -0.09360, -0.40125),
    vec3(-0.22704, 0.20398, -0.58602),
    vec3(-0.77324, 0.00241, -0.42203),
    vec3(0.04166, 0.16202, 0.41309),
    vec3(-0.15168, -0.27253, -0.40912),
    vec3(0.30875, 0.19126, -0.19523),
    vec3(-0.29932, -0.35644, 0.02675),
    vec3(0.13557, -0.23701, 0.62421),
    vec3(0.07220, -0.03664, -0.23095),
    vec3(0.13721, 0.14113, 0.19124),
    vec3(0.25400, -0.07378, -0.11854),
    vec3(2.47172, 2.25858, 1.28714),
    vec3(0.39278, 0.18506, 0.43350),
);

const NA_B3_MIE = vec3(0.00015, -0.00234, -0.00069);

struct NeuralAtmoSample {
    background: vec3<f32>,
    aerosols: vec3<f32>,
}

fn neural_atmosphere_eval(view_y: f32, light_y: f32, cosine: f32, az_cos: f32) -> NeuralAtmoSample {
    let vy = clamp(view_y, -0.05, 1.0);
    let ly = clamp(light_y, -0.35, 1.0);
    let x = array<f32, 6>(
        vy,
        sqrt(max(vy, 0.0)),
        ly,
        ly * abs(ly),
        cosine,
        az_cos,
    );
    var h1 = array<vec4<f32>, 4>(NA_B1[0], NA_B1[1], NA_B1[2], NA_B1[3]);
    for (var i = 0u; i < 6u; i += 1u) {
        let xi = x[i];
        h1[0] += NA_W1[i] * xi;
        h1[1] += NA_W1[6u + i] * xi;
        h1[2] += NA_W1[12u + i] * xi;
        h1[3] += NA_W1[18u + i] * xi;
    }
    h1[0] = max(h1[0], vec4(0.0));
    h1[1] = max(h1[1], vec4(0.0));
    h1[2] = max(h1[2], vec4(0.0));
    h1[3] = max(h1[3], vec4(0.0));

    var h2 = array<vec4<f32>, 4>(NA_B2[0], NA_B2[1], NA_B2[2], NA_B2[3]);
    for (var g = 0u; g < 4u; g += 1u) {
        let base = g * 4u;
        let v = h1[g];
        h2[0] += NA_W2[base] * v.x + NA_W2[base + 1u] * v.y + NA_W2[base + 2u] * v.z + NA_W2[base + 3u] * v.w;
        h2[1] += NA_W2[16u + base] * v.x + NA_W2[17u + base] * v.y + NA_W2[18u + base] * v.z + NA_W2[19u + base] * v.w;
        h2[2] += NA_W2[32u + base] * v.x + NA_W2[33u + base] * v.y + NA_W2[34u + base] * v.z + NA_W2[35u + base] * v.w;
        h2[3] += NA_W2[48u + base] * v.x + NA_W2[49u + base] * v.y + NA_W2[50u + base] * v.z + NA_W2[51u + base] * v.w;
    }
    h2[0] = max(h2[0], vec4(0.0));
    h2[1] = max(h2[1], vec4(0.0));
    h2[2] = max(h2[2], vec4(0.0));
    h2[3] = max(h2[3], vec4(0.0));

    var out_bg = NA_B3_BG;
    var out_mie = NA_B3_MIE;
    for (var g = 0u; g < 4u; g += 1u) {
        let base = g * 4u;
        let v = h2[g];
        out_bg += NA_W3_BG[base] * v.x + NA_W3_BG[base + 1u] * v.y + NA_W3_BG[base + 2u] * v.z + NA_W3_BG[base + 3u] * v.w;
        out_mie += NA_W3_MIE[base] * v.x + NA_W3_MIE[base + 1u] * v.y + NA_W3_MIE[base + 2u] * v.z + NA_W3_MIE[base + 3u] * v.w;
    }
    let bg = max(exp(max(out_bg, vec3(0.0))) - vec3(1.0), vec3(0.0)) * 0.04;
    let mie = max(exp(max(out_mie, vec3(0.0))) - vec3(1.0), vec3(0.0)) * 0.04;
    return NeuralAtmoSample(bg, mie);
}

fn fog_vapour_density(celsius: f32) -> f32 {
    return 216.7 * 6.1094 * exp(17.625 * celsius / (243.04 + celsius)) / (273.15 + celsius);
}

fn decode_scattering_half(bits: u32) -> f32 {
    let exponent = (bits >> 10u) & 31u;
    let mantissa = bits & 1023u;
    let sign = (bits & 32768u) << 16u;
    let normal = bitcast<f32>(sign | ((exponent + 112u) << 23u) | (mantissa << 13u));
    let subnormal = bitcast<f32>(sign | bitcast<u32>(f32(mantissa) * (1.0 / 16777216.0)));
    return select(normal, subnormal, exponent == 0u);
}

// Photon uses Bruneton's coordinates for two RGB16F scattering banks.
// The PNG stores the first four bytes at x and the last two at x + 32.
fn scattering_texel(cell: vec3<i32>) -> vec3<f32> {
    let pixel = vec2(cell.x, cell.y + 64 * cell.z);
    let rg = vec4<u32>(round(textureLoad(scattering_atlas, pixel, 0) * 255.0));
    let b = vec2<u32>(round(textureLoad(scattering_atlas, pixel + vec2(32, 0), 0).rg * 255.0));
    return vec3(
        decode_scattering_half(rg.x | (rg.y << 8u)),
        decode_scattering_half(rg.z | (rg.w << 8u)),
        decode_scattering_half(b.x | (b.y << 8u)),
    );
}

fn scattering_lookup(coordinates: vec3<f32>, bank: i32) -> vec3<f32> {
    let texel = clamp(coordinates, vec3(0.0), vec3(15.0, 63.0, 31.0));
    let lo = vec3<i32>(floor(texel));
    let hi = min(lo + 1, vec3(15, 63, 31));
    let f = fract(texel);
    let x0 = lo.x + bank;
    let x1 = hi.x + bank;

    let front_lo = mix(scattering_texel(vec3(x0, lo.y, lo.z)), scattering_texel(vec3(x1, lo.y, lo.z)), f.x);
    let front_hi = mix(scattering_texel(vec3(x0, hi.y, lo.z)), scattering_texel(vec3(x1, hi.y, lo.z)), f.x);
    let back_lo = mix(scattering_texel(vec3(x0, lo.y, hi.z)), scattering_texel(vec3(x1, lo.y, hi.z)), f.x);
    let back_hi = mix(scattering_texel(vec3(x0, hi.y, hi.z)), scattering_texel(vec3(x1, hi.y, hi.z)), f.x);
    return mix(mix(front_lo, front_hi, f.y), mix(back_lo, back_hi, f.y), f.z);
}

// Distances and radii are in kilometres.
const PLANET_RADIUS: f32 = 6371.0;
const INNER_RADIUS: f32 = 6370.0;
const OUTER_RADIUS: f32 = 6481.0;
const HORIZON_SQUARED: f32 = (OUTER_RADIUS - INNER_RADIUS) * (OUTER_RADIUS + INNER_RADIUS);
const OBSERVER_SQUARED: f32 = (PLANET_RADIUS - INNER_RADIUS) * (PLANET_RADIUS + INNER_RADIUS);

fn exit_distance(altitude: f32) -> f32 {
    let radial = INNER_RADIUS * altitude;
    return sqrt(radial * radial + HORIZON_SQUARED) - radial;
}

fn coordinates_for_scattering(cosine: f32, view_y: f32, light_y: f32) -> vec3<f32> {
    let mid = view_y * light_y;
    let extent = sqrt(max((1.0 - view_y * view_y) * (1.0 - light_y * light_y), 0.0));
    let azimuth = clamp(0.5 + (cosine - mid) / max(2.0 * extent, 1e-5), 0.0, 1.0);

    // Photon's positive-elevation remap makes twilight visible above the horizon.
    var mu = view_y;
    if mu > 0.0 {
        mu *= sqrt(sqrt(mu));
    }
    let radial = PLANET_RADIUS * mu;
    let ground = radial * radial - OBSERVER_SQUARED;
    let horizon = sqrt(HORIZON_SQUARED);
    let rho = sqrt(OBSERVER_SQUARED);
    let sky_min = OUTER_RADIUS - PLANET_RADIUS;
    var view_index = 32.0 + 31.0 * clamp((sqrt(max(ground + HORIZON_SQUARED, 0.0)) - radial - sky_min)
        / (rho + horizon - sky_min), 0.0, 1.0);
    if mu < 0.0 && ground >= 0.0 {
        let distance = -radial - sqrt(ground);
        view_index = 31.0 - 31.0 * clamp((distance - 1.0) / (rho - 1.0), 0.0, 1.0);
    }

    let thickness = OUTER_RADIUS - INNER_RADIUS;
    let a = (exit_distance(light_y) - thickness) / (horizon - thickness);
    let range = (exit_distance(-0.35) - thickness) / (horizon - thickness);
    let light_index = 31.0 * clamp((1.0 - a / range) / (1.0 + a), 0.0, 1.0);
    return vec3(azimuth * 15.0, view_index, light_index);
}

// Jendersie & d'Eon (SIGGRAPH 2023) analytic Draine + HG aerosol phase modulated
// by smooth regional humidity (no raw biome downfall so sky never pops at borders).
fn mie_phase(cosine: f32, g: f32) -> f32 {
    let hum = clamp(0.42 + 0.28 * sin(2.0 * weather_cycle() + 0.7) + 0.38 * frame.wetness, 0.0, 1.0);
    let g_eff = clamp(mix(g + 0.03, g - 0.06, hum * 0.7 + frame.rain * 0.3), 0.55, 0.93);
    let gg = g_eff * g_eff;
    let d_hg = max(1.0 + gg - 2.0 * g_eff * cosine, 0.0001);
    let hg = (1.0 - gg) / (2.0 * TAU * d_hg * sqrt(d_hg));
    let g_d = 0.62;
    let gd2 = g_d * g_d;
    let a_d = mix(8.0, 18.0, hum);
    let d_dr = max(1.0 + gd2 - 2.0 * g_d * cosine, 0.0001);
    let draine = ((1.0 - gd2) / (2.0 * TAU * d_dr * sqrt(d_dr)))
        * (1.0 + a_d * cosine * cosine) / (1.0 + a_d * (1.0 + 2.0 * gd2) * (1.0 / 3.0));
    return mix(hg, draine, 0.28 + 0.14 * hum);
}

fn scattering_from_light(direction: vec3<f32>, light: vec3<f32>, g: f32, mie_scale: f32) -> vec3<f32> {
    let mu = max(direction.y, mix(-0.009, 0.027, smoothstep(-0.05, 0.1, frame.sun_direction.y)));
    let cosine = clamp(dot(direction, light), -1.0, 1.0);
    if neural_atmosphere {
        let extent = sqrt(max((1.0 - mu * mu) * (1.0 - light.y * light.y), 0.0));
        let az_cos = clamp((cosine - mu * light.y) / max(extent, 1e-4), -1.0, 1.0);
        let spec = neural_atmosphere_eval(mu, light.y, cosine, az_cos);
        return spec.background + spec.aerosols * (mie_phase(cosine, g) * mie_scale);
    }
    let xyz = coordinates_for_scattering(cosine, mu, light.y);
    let background = scattering_lookup(xyz, 0);
    let aerosols = scattering_lookup(xyz, 16);
    return background + aerosols * (mie_phase(cosine, g) * mie_scale);
}

fn to_bt709(color: vec3<f32>) -> vec3<f32> {
    return max(REC2020_TO_REC709 * color, vec3(0.0));
}

fn sky_radiance(direction: vec3<f32>) -> vec3<f32> {
    let solar_elevation = frame.sun_direction.y;
    let twilight = exp(-190.0 * (solar_elevation + 0.096) * (solar_elevation + 0.096));
    var radiance = vec3(0.0);
    if solar_elevation > -0.35 {
        let solar_intensity = 7.0 * (1.0 + 6.0 * twilight) * smoothstep(-0.35, -0.22, solar_elevation);
        radiance += scattering_from_light(direction, frame.sun_direction, 0.86, 0.2) * solar_intensity;
    }
    let night = 1.0 - smoothstep(-0.10, 0.04, solar_elevation);
    if night > 0.0 && frame.moon_direction.y > -0.15 {
        radiance += scattering_from_light(direction, frame.moon_direction, 0.82, 0.5)
            * vec3(0.23, 0.32, 0.48) * (night * lunar_brightness());
    }
    let elev = max(direction.y, 0.0);
    // Physical civil twilight optics (Earth's Shadow, Belt of Venus, Chappuis Ozone zenith).
    let civil_twilight = smoothstep(-0.15, -0.02, solar_elevation) * (1.0 - smoothstep(0.01, 0.11, solar_elevation))
        * (1.0 - 0.85 * frame.rain);
    let anti_sun = clamp(-dot(normalize(vec3(direction.x, 0.0, direction.z) + vec3(1e-5, 0.0, 0.0)),
        normalize(vec3(frame.sun_direction.x, 0.0, frame.sun_direction.z) + vec3(1e-5, 0.0, 0.0))), 0.0, 1.0);
    let anti_lobe = anti_sun * anti_sun;
    // Earth's shadow rises from the anti-solar horizon as the sun sinks below 0 deg.
    let shadow_rise = clamp(0.03 + max(-solar_elevation, 0.0) * 1.15, 0.03, 0.16);
    let earth_shadow = (1.0 - smoothstep(0.0, shadow_rise, elev)) * anti_lobe * civil_twilight;
    // Belt of Venus: warm rose-magenta backscattered arch riding right above Earth's shadow.
    let venus_band = smoothstep(shadow_rise * 0.45, shadow_rise * 1.15, elev)
        * (1.0 - smoothstep(shadow_rise * 1.25, shadow_rise + 0.19, elev)) * anti_lobe * civil_twilight;
    // Stratospheric Chappuis band (600 nm ozone absorption) deepens the overhead twilight zenith.
    let ozone_zenith = smoothstep(0.18, 0.78, elev) * civil_twilight;

    var sky_color = to_bt709(radiance);
    sky_color *= mix(vec3(1.0), vec3(0.46, 0.54, 0.72), earth_shadow * 0.75);
    sky_color *= mix(vec3(1.0), vec3(0.84, 0.86, 1.08), ozone_zenith * 0.45);
    sky_color += vec3(0.19, 0.085, 0.125) * (venus_band * (0.45 + 0.55 * (1.0 - frame.rain)));

    // Mesospheric night airglow (van Rhijn limb brightening peaking 5-18 deg above horizon).
    let limb = smoothstep(0.0, 0.08, elev) * (1.0 - smoothstep(0.16, 0.65, elev));
    let airglow = (vec3(0.0024, 0.0044, 0.011) + vec3(0.0028, 0.0058, 0.0068) * limb)
        * (0.55 + 0.45 * night) * (1.0 - 0.55 * frame.rain);
    // Localized directional lightning scatter in the sky dome toward the active storm cell.
    let flash_align = 0.25 + 0.75 * pow(clamp(dot(direction, storm_flash_dir()) * 0.5 + 0.5, 0.0, 1.0), 3.0);
    sky_color += airglow + storm_flash() * vec3(0.44, 0.63, 0.95) * (0.24 * flash_align);

    let grey = dot(sky_color, frame.luminance);
    return mix(sky_color, vec3(grey) * vec3(0.82, 0.91, 1.04), frame.rain * 0.67);
}

fn optical_airmass(elevation: f32, scale_height: f32) -> f32 {
    let horizon_mass = sqrt(0.25 * TAU * PLANET_RADIUS / scale_height);
    return scale_height * horizon_mass / (1.0 + (horizon_mass - 1.0) * max(elevation, 0.0));
}

fn direct_light(light_y: f32, tint: vec3<f32>) -> vec3<f32> {
    let molecules = optical_airmass(light_y, 8.4);
    let aerosols = optical_airmass(light_y, 1.25);
    let extinction = vec3(0.0089, 0.0180, 0.0409) * molecules
        + vec3(0.00167, 0.00181, 0.00196) * aerosols;
    return tint * exp(-extinction) * smoothstep(-0.015, 0.055, light_y) * (1.0 - 0.68 * frame.rain);
}

// Photon's incident light is in irradiance units, for terrain's 1/pi diffuse
// response and the volume phase functions. Use Horizon's actual shadow light.
fn photon_incident_light() -> vec3<f32> {
    let light_y = frame.light_direction.y;
    let sun_y = frame.sun_direction.y;
    let daylight = photon_daylight();
    let blue_offset = sun_y + 0.09604;
    let blue_hour = clamp((exp(-190.0 * blue_offset * blue_offset) - 0.05) / 0.95, 0.0, 1.0);
    let tint_weight = 1.0 - smoothstep(0.0, 0.40, abs(sun_y - 0.17));
    let twilight_tint = mix(vec3(1.0), vec3(1.05, 0.84, 0.93) * 1.2, tint_weight * tint_weight);
    let solar_tint = twilight_tint * mix(vec3(1.0), vec3(0.95, 0.80, 1.0), blue_hour);
    let sun_exposure = 9.0 * (1.0 + 0.42 * (daylight.sunrise + daylight.sunset) + 34.0 * blue_hour);
    let moon_tint = REC709_TO_REC2020 * pow(vec3(0.75, 0.83, 1.0), vec3(2.2));
    let moon_exposure = 0.66 * (1.0 + 0.33 / clamp(1.25 * -sun_y, 0.125, 1.0));
    let uses_moon = dot(frame.light_direction, frame.moon_direction)
        > dot(frame.light_direction, frame.sun_direction);

    let rayleigh_ozone = vec3(0.00888980344, 0.01802700626, 0.04085573974);
    let mie = vec3(0.00166644236, 0.00181268513, 0.00195892790);
    let extinction = rayleigh_ozone * optical_airmass(light_y, 8.4) + mie * optical_airmass(light_y, 1.25);
    let light = vec3(1.051, 0.985, 0.940) * exp(-(REC709_TO_REC2020 * extinction));
    let grey = dot(light, vec3(0.2627, 0.6780, 0.0593));
    let twilight_fade = clamp(50.0 * light_y, 0.0, 1.0)
        * (0.75 + 0.25 * smoothstep(0.0, 0.11, abs(light_y - 0.15)));
    return to_bt709(mix(light, vec3(grey), 0.67 * frame.rain)
        * select(solar_tint * sun_exposure, moon_tint * (moon_exposure * lunar_brightness()), uses_moon)) * twilight_fade;
}

// Saturation vapor pressure over liquid water in hPa (Alduchov-Eskridge / Magnus-Tetens).
fn sat_vapor_pressure_hpa(temp_c: f32) -> f32 {
    return 6.1094 * exp(17.625 * temp_c / (243.04 + temp_c));
}

// Saturation water vapor density in g/m^3 at standard sea-level pressure (1013.25 hPa).
fn sat_vapor_density_gm3(temp_c: f32) -> f32 {
    let es = sat_vapor_pressure_hpa(temp_c);
    return (0.622 * es / max(1013.25 - 0.378 * es, 100.0)) * 1225.0;
}

// Per-forest-type micrometeorology fog model (evapotranspiration by forest class,
// diurnal radiative cooling, Magnus-Tetens saturation, Kohler terpene aerosol swelling,
// and Kunkel 1984 extinction). Directly smooths & rate-limits the final extinction
// via sky_lighting_previous (tau = 12s) so crossing biome borders never pops.
fn forest_fog() -> vec4<f32> {
    let prev = textureLoad(sky_lighting_previous, FOREST_FOG_CELL, 0);
    let valid_prev = frame.delta_time > 0.0 && frame.delta_time < 0.25 && prev.y > 1.0;

    let temp = frame.temperature;
    let down = frame.downfall;
    let skylight = frame.eye_light_smooth.y;
    let under_canopy = smoothstep(0.18, 0.50, skylight) * (1.0 - smoothstep(0.72, 0.98, skylight));

    // Smooth overlapping weights for Minecraft's major forest archetypes:
    let w_taiga = smoothstep(0.05, 0.24, temp) * (1.0 - smoothstep(0.32, 0.54, temp))
        * smoothstep(0.42, 0.76, down);
    let w_temperate = smoothstep(0.42, 0.62, temp) * (1.0 - smoothstep(0.76, 0.92, temp))
        * smoothstep(0.40, 0.72, down);
    let w_swamp = smoothstep(0.68, 0.82, temp) * (1.0 - smoothstep(0.86, 0.98, temp))
        * smoothstep(0.76, 0.92, down) * (1.0 - smoothstep(66.0, 82.0, frame.camera_position.y));
    let w_jungle = smoothstep(0.80, 0.95, temp) * (1.0 - smoothstep(1.10, 1.35, temp))
        * smoothstep(0.74, 0.92, down);
    let w_canopy = under_canopy * smoothstep(0.30, 0.62, down);

    // Total forest cover (clamped 0..1); deserts/savannas (downfall ~ 0) stay 0.
    let forest_cover = clamp(
        0.72 * w_taiga + 0.62 * w_temperate + 0.88 * w_swamp + 0.92 * w_jungle + 0.35 * w_canopy,
        0.0, 1.0,
    );
    // Species-specific evapotranspiration strength, BVOC terpene haze, and inversion height.
    let transpire = clamp(
        0.42 * w_taiga + 0.36 * w_temperate + 0.56 * w_swamp + 0.60 * w_jungle + 0.24 * w_canopy,
        0.0, 0.72,
    );
    let terpene_boost = 0.0014 * w_taiga + 0.0005 * w_jungle + 0.0003 * w_temperate;
    let target_cool_tint = clamp(0.85 * w_taiga + 0.20 * w_canopy - 0.15 * w_jungle, 0.0, 1.0);

    // Blend regional 8-day parcel temperature with biome temperature so narrow river
    // crossings inside a forest never cause sudden temperature/fog drops.
    let cycle = weather_cycle();
    let regional_c = 15.0 + 5.0 * cos(3.0 * cycle);
    let biome_c = clamp(temp * 22.0 + 1.0, -4.0, 34.0);
    let surface_c = mix(regional_c, biome_c, 0.55);

    // Diurnal longwave radiative cooling peaks at dawn; closed canopy slows morning burn-off.
    let night_age = fract(frame.day_time + 0.5);
    let burn_end = mix(0.65, 0.78, w_canopy);
    let rad_cooling = smoothstep(0.02, 0.48, night_age)
        * (1.0 - smoothstep(0.51, burn_end, night_age));
    let calm_air = 1.0 - smoothstep(0.20, 0.82, weather_wind() * (1.0 - 0.35 * w_canopy));
    let delta_t_cool = (6.6 + 1.4 * w_swamp) * rad_cooling * calm_air * (1.0 - 0.65 * frame.rain);

    // Evapotranspiration + post-rain wet foliage evaporation narrow the dewpoint spread.
    let base_spread = mix(5.8, 2.2, weather_humidity()) * (1.0 - 0.55 * transpire);
    let wet_spread = base_spread * (1.0 - 0.80 * max(frame.wetness, frame.rain));
    let t_dew = surface_c - wet_spread;
    let t_air = surface_c - delta_t_cool;

    // Magnus-Tetens saturation & Clausius-Clapeyron condensed liquid water content.
    let e_dew = sat_vapor_pressure_hpa(t_dew);
    let e_air = sat_vapor_pressure_hpa(t_air);
    let rh = e_dew / max(e_air, 0.1);
    let excess_gm3 = max(0.0, sat_vapor_density_gm3(t_dew) - sat_vapor_density_gm3(t_air));
    let lwc = min(excess_gm3 * 0.14, 0.24);
    // Kunkel (1984) empirical fog extinction law: beta_ext = 0.1447 * LWC^0.88 (m^-1).
    let beta_cond = select(0.0, 0.1447 * pow(lwc, 0.88), lwc > 1e-4);

    // Kohler hygroscopic swelling of biogenic forest aerosols (terpenes) at 70..98% RH.
    let rh_clamped = clamp(rh, 0.0, 0.96);
    let kohler = smoothstep(0.70, 0.98, rh) * (inverseSqrt(max(1.0 - rh_clamped, 0.04)) - 1.0);
    let beta_haze = (0.00075 + terpene_boost) * kohler;

    // Post-rain sunlit forest & meadow steam: when rain ends and sun breaks through
    // over wet ground, warm evaporation raises a brief low-lying sunlit mist.
    let post_rain = smoothstep(0.14, 0.55, frame.wetness) * (1.0 - smoothstep(0.04, 0.25, frame.rain))
        * smoothstep(0.06, 0.30, frame.sun_direction.y) * smoothstep(0.25, 0.65, down);
    let beta_steam = 0.0028 * post_rain * (0.45 + 0.55 * forest_cover);

    // Burn off midday forest haze under bright direct sun while keeping dawn/dusk/wet mist rich.
    let noon_burnoff = mix(1.0, 0.18, photon_daylight().noon * (1.0 - frame.rain));
    let target_ext = clamp((beta_cond * 0.35 + beta_haze * noon_burnoff) * forest_cover + beta_steam, 0.0, 0.011);
    let base_h = mix(15.0, 20.0, w_taiga);
    let species_h = mix(mix(base_h, 28.0, w_jungle), 11.0, w_swamp);
    let target_h = mix(mix(species_h, max(species_h, 24.0), max(frame.wetness, frame.rain)), 13.0, post_rain * 0.55);

    if !valid_prev {
        return vec4(target_ext, target_h, target_cool_tint, 1.0);
    }
    // Smooth + rate-limit the final output directly (tau = 12s, max +/-0.0014/s)
    // so crossing any biome border is completely seamless even when flying fast.
    let alpha = 1.0 - exp(-frame.delta_time / 12.0);
    let raw_ext = mix(prev.x, target_ext, alpha);
    let max_step = 0.0014 * frame.delta_time;
    let smooth_ext = prev.x + clamp(raw_ext - prev.x, -max_step, max_step);
    let smooth_h = mix(prev.y, target_h, alpha);
    let smooth_tint = mix(prev.z, target_cool_tint, alpha);
    return vec4(smooth_ext, smooth_h, smooth_tint, 1.0);
}

fn compute(input: ComputeInput) {
    if !input.in_bounds {
        return;
    }
    let cell = vec2<i32>(input.cell.xy);
    if all(cell == EXPOSURE_CELL) {
        // True screen-metered eye adaptation (auto-exposure) using a center-weighted
        // 5x5 log-average luminance of the un-adapted previous frame (taa_history_previous,
        // written before auto_exp is applied, so there is zero feedback pumping).
        // Asymmetric human pupil kinetics: fast constriction in bright light (tau = 0.55s),
        // slower dark adaptation entering caves/night (tau = 1.90s). Works in all dimensions.
        let hist_size = vec2<i32>(textureDimensions(taa_history_previous));
        let hist_f = vec2<f32>(hist_size);
        var log_sum = 0.0;
        var w_sum = 0.0;
        if hist_size.x > 4 && hist_size.y > 4 {
            for (var iy = 0; iy < 5; iy += 1) {
                let ny = (f32(iy) - 2.0) * 0.5;
                let v = 0.5 + ny * 0.68;
                for (var ix = 0; ix < 5; ix += 1) {
                    let nx = (f32(ix) - 2.0) * 0.5;
                    let u = 0.5 + nx * 0.68;
                    let px = clamp(vec2<i32>(vec2(u, v) * hist_f), vec2(0), hist_size - 1);
                    let rgb = max(textureLoad(taa_history_previous, px, 0).rgb, vec3(0.0));
                    let luma = dot(rgb, frame.luminance);
                    let w = exp(-(nx * nx + ny * ny) * 2.0);
                    log_sum += clamp(log2(max(luma, 0.002)), -8.5, 4.5) * w;
                    w_sum += w;
                }
            }
        }
        let avg_luma = select(0.42, exp2(log_sum / max(w_sum, 1e-4)), w_sum > 0.0);
        let target_exp = clamp(pow(0.42 / max(avg_luma, 0.005), 0.22), 0.85, 1.38);
        let prev_exp = textureLoad(sky_lighting_previous, EXPOSURE_CELL, 0).x;
        let valid_exp = frame.delta_time > 0.0 && frame.delta_time < 0.25 && prev_exp > 0.2;
        var adapted = target_exp;
        if valid_exp {
            let tau = select(1.90, 0.55, target_exp < prev_exp);
            let rate = 1.0 - exp(-frame.delta_time / tau);
            adapted = mix(prev_exp, target_exp, rate);
        }
        textureStore(sky_lighting, cell, vec4(adapted, avg_luma, 0.0, 1.0));
        return;
    }
    if frame.dimension != 0 {
        textureStore(sky_lighting, cell, vec4(0.0));
        return;
    }

    if cell.y == SKY_LIGHT_ROW {
        if cell.x == 0 {
            textureStore(sky_lighting, cell, vec4(photon_incident_light() + storm_flash() * vec3(0.42, 0.6, 1.0), 1.0));
        } else if cell.x == 1 {
            let opposite_sun = normalize(vec3(-frame.sun_direction.x, 0.8, -frame.sun_direction.z));
            let dome = (sky_radiance(vec3(0.0, 1.0, 0.0)) + sky_radiance(opposite_sun)) * 0.5;
            // Match upstream Photon's SH skylight irradiance (~pi * L_sky) in bright daylight
            // so shadows stay deep and crisp while twilight/night keep full visibility.
            let day_amb_scale = mix(2.25, 1.22, smoothstep(-0.04, 0.25, frame.sun_direction.y));
            textureStore(sky_lighting, cell, vec4(dome * day_amb_scale + aurora_amount() * vec3(0.012, 0.046, 0.029), 1.0));
        } else if cell.x == 2 {
            let sun = direct_light(frame.sun_direction.y, vec3(1.95, 1.83, 1.72));
            textureStore(sky_lighting, cell, vec4(sun, 1.0));
        } else if all(cell == FOREST_FOG_CELL) {
            textureStore(sky_lighting, cell, forest_fog());
        } else if all(cell == CLOUD_WEATHER_CELL) {
            // Idealized regional parcel. Temperatures are Celsius and one block is one metre.
            // Swing surface_c (13..29 C) and dewpoint_spread (4.5..23.5 C) across the 8-day
            // cycle so clear/scattered blue-sky days (~50%), partly cloudy days (~25%), and
            // moody overcast days (~25%) alternate naturally instead of staying overcast.
            let cycle = weather_cycle();
            let surface_c = 21.0 + 8.0 * cos(3.0 * cycle) - 4.0 * photon_daylight().midnight;
            let dewpoint_spread = (14.0 + 9.5 * sin(2.0 * cycle + 0.7)) * (1.0 - frame.wetness);
            // Lift from sea level (63) to the low cloud base (183). Before saturation,
            // temperature falls about 9.8 K/km and dew point about 2 K/km.
            let lift_km = 0.120;
            let dry_lift = min(lift_km, dewpoint_spread / (9.8 - 2.0));
            let air_c = surface_c - 9.8 * dry_lift - 6.0 * (lift_km - dry_lift);
            let dew_c = min(air_c, surface_c - dewpoint_spread - 2.0 * dry_lift);
            // Ratio of saturation vapour pressures over liquid water (Magnus formula).
            let rh = exp(17.625 * (dew_c / (243.04 + dew_c) - air_c / (243.04 + air_c)));
            // Keep CLOUD_WEATHER_CELL.zw zeroed so it doesn't stack a 40-block morning
            // fog wall on top of our per-forest-type valley mist and aerial perspective.
            textureStore(sky_lighting, cell, vec4(clamp(air_c / 32.0, 0.0, 1.0), rh, 0.0, 16.0));
        }
        return;
    }

    // The two guard columns wrap the longitude seam under bilinear sampling.
    let azimuth = (fract((f32(cell.x) - 0.5) / f32(SKY_AZIMUTH_TEXELS)) - 0.5) * TAU;
    let height = 2.0 * f32(cell.y) / f32(SKY_ELEVATION_TEXELS - 1u) - 1.0;
    let vertical = height * abs(height);
    let horizontal = sqrt(max(1.0 - vertical * vertical, 0.0));
    let direction = vec3(sin(azimuth) * horizontal, vertical, cos(azimuth) * horizontal);
    textureStore(sky_lighting, cell, vec4(sky_radiance(direction), 1.0));
}
