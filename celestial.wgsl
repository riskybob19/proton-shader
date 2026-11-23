override galaxy_enabled: bool = false;
override end_sun: bool = true;

fn celestial_transmittance(elevation: f32) -> vec3<f32> {
    let air_mass = 1.0 / (0.055 + max(0.0, elevation));
    return exp(-vec3(0.045, 0.09, 0.19) * air_mass);
}

fn sky_turn(ray: vec3<f32>) -> vec3<f32> {
    let angle = TAU * frame.day_time;
    let c = cos(angle);
    let s = sin(angle);
    return vec3(ray.x * c + ray.y * s, ray.y * c - ray.x * s, ray.z);
}

fn star_cell(cell: vec2<i32>, density: f32, airmass: f32) -> vec3<f32> {
    let key = cloud_hash(cell);
    let threshold = 1.0 - density;
    if key <= threshold {
        return vec3(0.0);
    }
    let u = clamp((key - threshold) / max(density, 1e-4), 0.0, 1.0);
    let u2 = u * u;
    let intensity = u2 * u2;
    // Atmospheric scintillation: stars near the horizon twinkle faster/stronger through thicker air.
    let phase = key * 37.0 + frame.time * (2.2 + 2.8 * key);
    let twinkle = 1.0 + clamp(0.14 * airmass, 0.0, 0.42) * sin(phase) * cos(phase * 0.63 + 1.1);
    return intensity * twinkle * mix(vec3(0.74, 0.84, 1.0), vec3(1.0, 0.77, 0.52),
        cloud_hash(cell + vec2(63, 29)));
}

fn star_field(ray: vec3<f32>, density: f32) -> vec3<f32> {
    let airmass = clamp(0.65 / (0.16 + max(ray.y, 0.0)), 0.5, 3.0);
    let p = (ray.xy / max(0.01, abs(ray.z) + length(ray.xy))
        + vec2(select(-41.21, 41.21, ray.z >= 0.0))) * 365.0;
    let cell = vec2<i32>(floor(p));
    let frac = smoothstep(vec2(0.0), vec2(1.0), fract(p));
    let a = mix(star_cell(cell, density, airmass), star_cell(cell + vec2(1, 0), density, airmass), frac.x);
    let b = mix(star_cell(cell + vec2(0, 1), density, airmass), star_cell(cell + vec2(1, 1), density, airmass), frac.x);
    return mix(a, b, frac.y);
}

fn night_background(ray: vec3<f32>) -> vec3<f32> {
    let night = smoothstep(0.06, 0.28, -frame.sun_direction.y)
        * smoothstep(-0.08, 0.07, ray.y) * (1.0 - frame.rain);
    if night <= 0.001 {
        return vec3(0.0);
    }
    let rotated = sky_turn(ray);
    var galaxy = vec3(0.0);
    if galaxy_enabled {
        let uv = vec2(0.5 + atan2(rotated.x, rotated.z) / TAU,
            acos(clamp(-rotated.y, -1.0, 1.0)) / PHOTON_PI);
        galaxy = pow(textureSampleLevel(galaxy_atlas, galaxy_atlas_sampler, uv, 0.0).rgb, vec3(2.2))
            * vec3(0.20, 0.16, 0.27);
    }
    let density = 0.018 + 0.025 * clamp(dot(galaxy, frame.luminance), 0.0, 1.0);
    return night * celestial_transmittance(ray.y) * (galaxy + star_field(rotated, density) * 0.8);
}

fn sun_disc(ray: vec3<f32>) -> vec3<f32> {
    let cosine = dot(ray, frame.sun_direction);
    if cosine < 0.99973 {
        return vec3(0.0);
    }
    let radial = clamp((1.0 - cosine) * (1.0 / 0.00024), 0.0, 1.0);
    let edge = smoothstep(0.99973, 0.99980, cosine);
    let limb = pow(vec3(max(0.015, 1.0 - radial)), vec3(0.22, 0.26, 0.31));
    return solar_atmosphere_light() * limb * (edge * 11.0 * (1.0 - frame.rain));
}

fn moon_disc(ray: vec3<f32>) -> vec4<f32> {
    let moon = frame.moon_direction;
    let cosine = dot(ray, moon);
    if cosine < 0.99969 || ray.y < 0.0 {
        return vec4(0.0);
    }
    let axis = select(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), abs(moon.y) > 0.98);
    let tangent = normalize(cross(axis, moon));
    let up = cross(moon, tangent);
    let plane = vec2(dot(ray, tangent), dot(ray, up)) / 0.024;
    let top = sqrt(max(0.0, 1.0 - dot(plane, plane)));
    let phase = TAU / 8.0 * f32(frame.moon_phase);
    let lit = photon_dampen(max(0.0, -sin(phase) * plane.x + cos(phase) * top));
    let tex = textureSampleLevel(wave_noise, wave_noise_sampler, 0.5 + plane * 0.19, 0.0).rg;
    let craters = 0.53 + 0.31 * tex.r + 0.16 * tex.g;
    let edge = smoothstep(0.99969, 0.99974, cosine) * (1.0 - frame.rain);
    let tint = vec3(0.63, 0.73, 0.92) * celestial_transmittance(ray.y);
    return vec4(tint * (0.009 + 1.55 * lit * craters) * edge, edge);
}

// Physical 22-degree hexagonal ice-crystal refraction halo around the moon (and subtle
// solar parhelic ring) in cool/cirrus skies (uses regional temperature, strict low < high smoothstep).
fn ice_crystal_halo(ray: vec3<f32>) -> vec3<f32> {
    if ray.y <= 0.0 || frame.rain >= 0.9 {
        return vec3(0.0);
    }
    let reg_temp = cloud_weather().x;
    let cold_or_cirrus = clamp(
        (1.0 - smoothstep(0.25, 0.62, reg_temp)) * 0.80
        + smoothstep(0.30, 0.70, cloud_cover) * 0.40,
        0.0, 1.0,
    ) * (1.0 - frame.rain) * smoothstep(0.0, 0.12, ray.y);
    if cold_or_cirrus <= 0.01 {
        return vec3(0.0);
    }
    var halo = vec3(0.0);
    // Lunar 22-deg halo (cos(22 deg) = 0.9272) with crisp inner edge and soft outer dispersion.
    if frame.moon_direction.y > 0.08 {
        let cm = dot(ray, frame.moon_direction);
        let dm = cm - 0.9272;
        if abs(dm) < 0.08 {
            let inner_cut = 1.0 - smoothstep(0.925, 0.955, cm);
            let ring = exp(-dm * dm * 2600.0) * inner_cut;
            let tint = mix(vec3(0.72, 0.82, 1.0), vec3(1.0, 0.86, 0.72), smoothstep(-0.025, 0.015, dm));
            let night_vis = 1.0 - smoothstep(-0.05, 0.08, frame.sun_direction.y);
            halo += tint * (ring * 0.032 * lunar_brightness() * night_vis);
        }
    }
    // Subtle solar 22-deg cirrus halo.
    if frame.sun_direction.y > 0.12 {
        let cs = dot(ray, frame.sun_direction);
        let ds = cs - 0.9272;
        if abs(ds) < 0.08 {
            let inner_cut = 1.0 - smoothstep(0.924, 0.952, cs);
            let ring = exp(-ds * ds * 3200.0) * inner_cut;
            let tint = mix(vec3(0.84, 0.92, 1.0), vec3(1.0, 0.85, 0.70), smoothstep(-0.02, 0.015, ds));
            halo += solar_atmosphere_light() * tint * (ring * 0.022);
        }
    }
    return halo * cold_or_cirrus;
}

fn sky_celestial(ray: vec3<f32>) -> vec3<f32> {
    let moon = moon_disc(ray);
    return sun_disc(ray) + night_background(ray) * (1.0 - moon.a) + moon.rgb + ice_crystal_halo(ray);
}

fn end_sky(ray: vec3<f32>, vanilla_sky: vec3<f32>) -> vec3<f32> {
    let stars = star_field(ray, 0.006).xzy * smoothstep(-0.2, 0.18, ray.y) * 0.09;
    let direction = normalize(vec3(-0.83, 0.31, 0.46));
    let cosine = clamp(dot(direction, ray), -1.0, 1.0);
    let gradient = 1.0 - smoothstep(0.0, 0.65, abs(ray.y));

    // Layered 3D-projected volumetric cosmic nebula clouds lit by the End sun (1 texture fetch).
    let plane_uv = ray.xz / (abs(ray.y) + 0.36);
    let t1 = frame.time * 0.0018;
    let rot_uv = mat2x2<f32>(vec2(0.866, -0.5), vec2(0.5, 0.866)) * (plane_uv * 0.14);
    let n1 = textureSampleLevel(wave_noise, wave_noise_sampler, rot_uv + vec2(t1, -t1 * 0.6), 0.0);
    let neb_shape = smoothstep(0.24, 0.86, n1.a * 0.62 + n1.b * 0.38);
    let band = exp(-abs(ray.y - 0.12 * sin(atan2(ray.z, ray.x) * 2.0)) * 3.2);
    let neb_density = neb_shape * (0.24 + 0.76 * band) * smoothstep(-0.35, 0.08, ray.y);
    let neb_tint = mix(vec3(0.085, 0.028, 0.155), vec3(0.032, 0.105, 0.165), n1.r);
    let sun_rim = vec3(0.28, 0.13, 0.21) * (photon_phase(cosine, 0.62) * smoothstep(0.25, 0.78, n1.g));
    let nebula = (neb_tint + sun_rim * select(0.25, 1.0, end_sun)) * neb_density * 0.42;

    if !end_sun {
        return vanilla_sky * (0.55 + 0.20 * gradient) + nebula + stars;
    }
    let tint = vec3(1.0, 0.48, 0.22);
    var sun = tint * (0.048 * photon_phase(cosine, 0.6));
    if cosine > 0.91 {
        let tangent = normalize(cross(vec3(0.0, 1.0, 0.0), direction));
        let square = vec2(dot(ray, tangent), dot(ray, cross(direction, tangent)));
        let radius = acos(clamp(cosine, -1.0, 1.0));
        let angle = atan2(square.y, square.x + 1e-8) / TAU + 0.5;
        let noise = textureSampleLevel(wave_noise, wave_noise_sampler,
            vec2(angle + 0.015 * frame.time - 0.33 * radius, radius - 0.025 * frame.time), 0.0).r;
        let disc = 1.0 - smoothstep(0.018, 0.024, radius);
        let flare = pow(noise, 5.0) * exp(-22.0 * max(0.0, radius - 0.024)) * (1.0 - disc);
        sun += tint * (6.0 * disc + 0.55 * flare);
    }
    return vanilla_sky * (0.52 + 0.22 * gradient) + nebula + stars + sun;
}

fn aurora_volume(ray: vec3<f32>) -> vec3<f32> {
    let amount = aurora_amount();
    if amount <= 0.001 || ray.y <= 0.06 {
        return vec3(0.0);
    }
    let start = max(0.0, (610.0 - frame.camera_position.y) / ray.y);
    let end = min(3400.0, (950.0 - frame.camera_position.y) / ray.y);
    if end <= start {
        return vec3(0.0);
    }
    var emission = vec3(0.0);
    let step_size = (end - start) / 14.0;
    for (var i = 0u; i < 14u; i += 1u) {
        let world = frame.camera_position + ray * (start + (f32(i) + 0.5) * step_size);
        let h = clamp((world.y - 610.0) / 340.0, 0.0, 1.0);
        let bend = sin(world.x * 0.008 + frame.time * 0.04) * 0.8
            + sin(world.x * 0.021 - frame.time * 0.025) * 0.17;
        let curtain = 1.0 - abs(sin(world.z * 0.025 + bend));
        let noise = textureSampleLevel(wave_noise, wave_noise_sampler,
            world.xz * vec2(0.012, 0.003) + vec2(0.0, frame.time * 0.002), 0.0).g;
        let density = pow(curtain, 13.0) * noise * smoothstep(0.0, 0.13, h) * (1.0 - h) * (1.0 - h);
        emission += mix(vec3(0.05, 0.60, 0.35), vec3(0.31, 0.08, 0.53), h) * density;
    }
    return emission * amount * step_size * 0.004 * smoothstep(0.06, 0.19, ray.y);
}
