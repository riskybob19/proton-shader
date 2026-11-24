// 0 = Live 8-Wavelength Spectral Hillaire (EGSR 2020 + fgarlin spectral matrix), 1 = Classic Bruneton LUT.
override atmosphere_model: u32 = 0u;

// 8-wavelength (400, 445, 490, 535 | 575, 610, 650, 700 nm) physical atmosphere cross-sections (km^-1):
//  - Bucholtz (1995) Rayleigh scattering (~lambda^-4.08, H_R = 8.4 km)
//  - Angstrom continental Mie aerosol extinction (~lambda^-0.85, H_M = 1.25 km, albedo = 0.90)
//  - Brion-Daumont-Malicet / Serdyuchenko Chappuis Ozone (O3, 300 DU, tent profile 10..40 km peak 25 km)
const SPEC_BETA_R_LO = vec4<f32>(0.045467, 0.029430, 0.019866, 0.013881);
const SPEC_BETA_R_HI = vec4<f32>(0.010343, 0.008127, 0.006272, 0.004636);
const SPEC_BETA_M_LO = vec4<f32>(0.001901, 0.001736, 0.001600, 0.001484);
const SPEC_BETA_M_HI = vec4<f32>(0.001396, 0.001328, 0.001258, 0.001181);
const SPEC_BETA_O_LO = vec4<f32>(0.000006, 0.000100, 0.000551, 0.001741);
const SPEC_BETA_O_HI = vec4<f32>(0.003019, 0.003100, 0.001534, 0.000551);

// ASTM E-490 AM0 solar spectrum * CIE 1931 2-deg CMFs -> Rec.2020 spectral integration matrix
// (Y-normalized so flat white reflectance under AM0 matches Photon sunlight_color in Rec.2020).
const SPEC_M2020_R_LO = vec4<f32>( 0.001406,  0.068290, -0.061497,  0.015809);
const SPEC_M2020_R_HI = vec4<f32>( 0.395517,  0.502760,  0.159548,  0.003162);
const SPEC_M2020_G_LO = vec4<f32>(-0.001598, -0.074240,  0.146421,  0.540786);
const SPEC_M2020_G_HI = vec4<f32>( 0.318119,  0.046376, -0.003993,  0.001255);
const SPEC_M2020_B_LO = vec4<f32>( 0.019298,  0.733166,  0.190559, -0.002975);
const SPEC_M2020_B_HI = vec4<f32>(-0.008163, -0.001224,  0.000106, -0.000033);

const SPEC_R_EARTH: f32 = 6371.0;
const SPEC_R_TOP: f32 = 6481.0;
const SPEC_H_R: f32 = 8.4;
const SPEC_H_M: f32 = 1.25;
const SPEC_MIE_ALBEDO: f32 = 0.90;

struct Spec8 {
    lo: vec4<f32>,
    hi: vec4<f32>,
}

struct SpectralAtmoSample {
    background: vec3<f32>,
    aerosols: vec3<f32>,
}

fn spec8_to_rec2020(s: Spec8) -> vec3<f32> {
    return max(vec3(
        dot(SPEC_M2020_R_LO, s.lo) + dot(SPEC_M2020_R_HI, s.hi),
        dot(SPEC_M2020_G_LO, s.lo) + dot(SPEC_M2020_G_HI, s.hi),
        dot(SPEC_M2020_B_LO, s.lo) + dot(SPEC_M2020_B_HI, s.hi),
    ), vec3(0.0));
}

fn spec_ray_sphere_exit(r: f32, mu: f32, radius: f32) -> f32 {
    let b = r * mu;
    let disc = b * b - (r * r - radius * radius);
    if disc < 0.0 {
        return 0.0;
    }
    return max(0.0, -b + sqrt(disc));
}

fn spec_hits_ground(r: f32, mu: f32) -> bool {
    let r_g = SPEC_R_EARTH - 0.5;
    return mu < 0.0 && (r * r * (1.0 - mu * mu)) < (r_g * r_g);
}

fn spec_densities(h: f32) -> vec3<f32> {
    let hc = max(h, 0.0);
    return vec3(
        exp(-hc * (1.0 / SPEC_H_R)),
        exp(-hc * (1.0 / SPEC_H_M)),
        max(0.0, 1.0 - abs(h - 25.0) * (1.0 / 15.0)),
    );
}

fn spec_solar_transmittance(r: f32, mu_s: f32, beta_m_lo: vec4<f32>, beta_m_hi: vec4<f32>) -> Spec8 {
    if spec_hits_ground(r, mu_s) {
        return Spec8(vec4(0.0), vec4(0.0));
    }
    let dist = spec_ray_sphere_exit(r, mu_s, SPEC_R_TOP);
    if dist <= 0.0 {
        return Spec8(vec4(1.0), vec4(1.0));
    }
    var tau = vec3(0.0);
    var t_prev = 0.0;
    for (var i = 1u; i <= 6u; i += 1u) {
        let u = f32(i) * (1.0 / 6.0);
        let t_cur = dist * (u * u);
        let dt = t_cur - t_prev;
        let tm = 0.5 * (t_prev + t_cur);
        t_prev = t_cur;
        let rp = sqrt(r * r + tm * tm + 2.0 * r * tm * mu_s);
        tau += spec_densities(rp - SPEC_R_EARTH) * dt;
    }
    let od_lo = SPEC_BETA_R_LO * tau.x + beta_m_lo * tau.y + SPEC_BETA_O_LO * tau.z;
    let od_hi = SPEC_BETA_R_HI * tau.x + beta_m_hi * tau.y + SPEC_BETA_O_HI * tau.z;
    return Spec8(exp(-od_lo), exp(-od_hi));
}

// Hillaire (EGSR 2020) spectral isotropic multi-scattering transfer function Psi_ms(h, mu_s)
// evaluated across all 8 wavelengths with stratospheric twilight upper-atmosphere illumination.
fn spec_hillaire_ms(r: f32, mu_s: f32, t_sun: Spec8,
    beta_m_lo: vec4<f32>, beta_m_hi: vec4<f32>,
    beta_ms_lo: vec4<f32>, beta_ms_hi: vec4<f32>) -> Spec8 {
    let h = max(r - SPEC_R_EARTH, 0.0);
    let er = exp(-h * (1.0 / SPEC_H_R));
    let em = exp(-h * (1.0 / SPEC_H_M));
    let o_up = max(0.0, 15.0 - max(h - 10.0, 0.0) * 0.5);

    let tau_up_lo = SPEC_BETA_R_LO * (SPEC_H_R * er) + beta_m_lo * (SPEC_H_M * em) + SPEC_BETA_O_LO * o_up;
    let tau_up_hi = SPEC_BETA_R_HI * (SPEC_H_R * er) + beta_m_hi * (SPEC_H_M * em) + SPEC_BETA_O_HI * o_up;
    let sca_up_lo = SPEC_BETA_R_LO * (SPEC_H_R * er) + beta_ms_lo * (SPEC_H_M * em);
    let sca_up_hi = SPEC_BETA_R_HI * (SPEC_H_R * er) + beta_ms_hi * (SPEC_H_M * em);
    let p_up_lo = (vec4(1.0) - exp(-2.0 * tau_up_lo)) * (sca_up_lo / max(tau_up_lo, vec4(1e-6)));
    let p_up_hi = (vec4(1.0) - exp(-2.0 * tau_up_hi)) * (sca_up_hi / max(tau_up_hi, vec4(1e-6)));

    let tau_dn_lo = SPEC_BETA_R_LO * (SPEC_H_R * (1.0 - er)) + beta_m_lo * (SPEC_H_M * (1.0 - em));
    let tau_dn_hi = SPEC_BETA_R_HI * (SPEC_H_R * (1.0 - er)) + beta_m_hi * (SPEC_H_M * (1.0 - em));
    let sca_dn_lo = SPEC_BETA_R_LO * (SPEC_H_R * (1.0 - er)) + beta_ms_lo * (SPEC_H_M * (1.0 - em));
    let sca_dn_hi = SPEC_BETA_R_HI * (SPEC_H_R * (1.0 - er)) + beta_ms_hi * (SPEC_H_M * (1.0 - em));
    let p_dn_lo = (vec4(1.0) - exp(-2.0 * tau_dn_lo)) * (sca_dn_lo / max(tau_dn_lo, vec4(1e-6)));
    let p_dn_hi = (vec4(1.0) - exp(-2.0 * tau_dn_hi)) * (sca_dn_hi / max(tau_dn_hi, vec4(1e-6)));

    let f_lo = clamp(0.5 * (p_up_lo + p_dn_lo + exp(-2.0 * tau_dn_lo) * 0.10), vec4(0.0), vec4(0.85));
    let f_hi = clamp(0.5 * (p_up_hi + p_dn_hi + exp(-2.0 * tau_dn_hi) * 0.10), vec4(0.0), vec4(0.85));

    // When the sun dips near/below the horizon, upper stratospheric air above Earth's shadow
    // still scatters down into lower twilight layers (producing natural blue-hour & afterglow).
    var t_eff_lo = t_sun.lo;
    var t_eff_hi = t_sun.hi;
    if mu_s < 0.08 {
        let r_sh = SPEC_R_EARTH / max(sqrt(max(1.0 - mu_s * mu_s, 1e-4)), 1e-4);
        let h_lit = max(h + 10.0, r_sh - SPEC_R_EARTH + 3.0);
        let t_up = spec_solar_transmittance(SPEC_R_EARTH + min(h_lit, 65.0), max(mu_s, -0.04), beta_m_lo, beta_m_hi);
        let att = exp(-max(h_lit - h, 0.0) * (1.0 / 14.0)) * 0.55;
        t_eff_lo = max(t_eff_lo, t_up.lo * att);
        t_eff_hi = max(t_eff_hi, t_up.hi * att);
    }

    let inv_4pi = 1.0 / (2.0 * TAU);
    return Spec8(
        t_eff_lo * (f_lo / (vec4(1.0) - f_lo)) * inv_4pi,
        t_eff_hi * (f_hi / (vec4(1.0) - f_hi)) * inv_4pi,
    );
}

// Live 8-wavelength Spectral Hillaire spherical-shell atmospheric integrator:
// real-time dynamic aerosols (humidity/rain), curved-ray sunset refraction, Chappuis ozone,
// and exact CIE 1931 -> Rec.2020 spectral projection (zero LUT quantization).
fn spectral_hillaire_eval(view_y: f32, light_y: f32, cosine: f32, steps: u32) -> SpectralAtmoSample {
    // Match Photon's horizon stretch so golden-hour gradients rise gracefully above the horizon.
    var mu = view_y;
    if mu > 0.0 {
        mu *= sqrt(sqrt(mu));
    }
    // Curved-ray atmospheric refraction near the horizon (~0.52 deg lift at sunset).
    let ly_eff = min(1.0, light_y + 0.009 * exp(-max(light_y + 0.03, 0.0) * 16.0));

    // Weather-reactive live Mie aerosols: crisp alpine sky in dry weather, richer haze in humid/wet air.
    let hum = clamp(0.42 + 0.28 * sin(2.0 * weather_cycle() + 0.7) + 0.38 * frame.wetness, 0.0, 1.0);
    let mie_mult = mix(0.78, 1.55, hum) * (1.0 + 0.35 * frame.rain);
    let beta_m_lo = SPEC_BETA_M_LO * mie_mult;
    let beta_m_hi = SPEC_BETA_M_HI * mie_mult;
    let beta_ms_lo = beta_m_lo * SPEC_MIE_ALBEDO;
    let beta_ms_hi = beta_m_hi * SPEC_MIE_ALBEDO;

    let r0 = SPEC_R_EARTH + 0.1;
    let dist = spec_ray_sphere_exit(r0, mu, SPEC_R_TOP);
    if dist <= 0.0 {
        return SpectralAtmoSample(vec3(0.0), vec3(0.0));
    }

    let sin_v = sqrt(max(1.0 - mu * mu, 0.0));
    let sin_s = sqrt(max(1.0 - ly_eff * ly_eff, 0.0));
    let cos_az = clamp((cosine - mu * ly_eff) / max(sin_v * sin_s, 1e-5), -1.0, 1.0);
    let ray_phase = (3.0 / (8.0 * TAU)) * (1.0 + cosine * cosine);

    var l_ray_lo = vec4(0.0);
    var l_ray_hi = vec4(0.0);
    var l_mie_lo = vec4(0.0);
    var l_mie_hi = vec4(0.0);
    var t_view_lo = vec4(1.0);
    var t_view_hi = vec4(1.0);

    let inv_steps = 1.0 / f32(steps);
    var t_prev = 0.0;
    for (var i = 1u; i <= steps; i += 1u) {
        let u = f32(i) * inv_steps;
        let t_cur = dist * (u * u);
        let dt = t_cur - t_prev;
        let tm = 0.5 * (t_prev + t_cur);
        t_prev = t_cur;

        let px = tm * sin_v;
        let py = r0 + tm * mu;
        let rp = sqrt(px * px + py * py);
        let mu_s_loc = (px * (sin_s * cos_az) + py * ly_eff) / rp;

        let d = spec_densities(rp - SPEC_R_EARTH);
        let ext_lo = SPEC_BETA_R_LO * d.x + beta_m_lo * d.y + SPEC_BETA_O_LO * d.z;
        let ext_hi = SPEC_BETA_R_HI * d.x + beta_m_hi * d.y + SPEC_BETA_O_HI * d.z;
        let step_t_lo = exp(-ext_lo * dt);
        let step_t_hi = exp(-ext_hi * dt);
        let integ_lo = (vec4(1.0) - step_t_lo) / max(ext_lo, vec4(1e-7));
        let integ_hi = (vec4(1.0) - step_t_hi) / max(ext_hi, vec4(1e-7));

        let t_sun = spec_solar_transmittance(rp, mu_s_loc, beta_m_lo, beta_m_hi);
        let psi = spec_hillaire_ms(rp, mu_s_loc, t_sun, beta_m_lo, beta_m_hi, beta_ms_lo, beta_ms_hi);

        let w_lo = t_view_lo * integ_lo;
        let w_hi = t_view_hi * integ_hi;
        l_ray_lo += w_lo * (SPEC_BETA_R_LO * d.x * (t_sun.lo * ray_phase + psi.lo) + beta_ms_lo * d.y * psi.lo);
        l_ray_hi += w_hi * (SPEC_BETA_R_HI * d.x * (t_sun.hi * ray_phase + psi.hi) + beta_ms_hi * d.y * psi.hi);
        l_mie_lo += w_lo * (beta_ms_lo * d.y * t_sun.lo);
        l_mie_hi += w_hi * (beta_ms_hi * d.y * t_sun.hi);

        t_view_lo *= step_t_lo;
        t_view_hi *= step_t_hi;
    }

    return SpectralAtmoSample(
        spec8_to_rec2020(Spec8(l_ray_lo, l_ray_hi)),
        spec8_to_rec2020(Spec8(l_mie_lo, l_mie_hi)),
    );
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

fn scattering_from_light(direction: vec3<f32>, light: vec3<f32>, g: f32, mie_scale: f32, steps: u32) -> vec3<f32> {
    let mu = max(direction.y, mix(-0.009, 0.027, smoothstep(-0.05, 0.1, frame.sun_direction.y)));
    let cosine = clamp(dot(direction, light), -1.0, 1.0);
    if atmosphere_model == 0u {
        let eff_steps = select(steps, min(steps, 12u), direction.y < -0.02);
        let spec = spectral_hillaire_eval(mu, light.y, cosine, eff_steps);
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
        radiance += scattering_from_light(direction, frame.sun_direction, 0.86, 0.2, 20u) * solar_intensity;
    }
    let night = 1.0 - smoothstep(-0.10, 0.04, solar_elevation);
    if night > 0.0 && frame.moon_direction.y > -0.15 {
        radiance += scattering_from_light(direction, frame.moon_direction, 0.82, 0.5, 10u)
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
