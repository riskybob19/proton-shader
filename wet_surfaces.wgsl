override wet_surfaces: bool = true;

struct WetSurface {
    darkening: f32,
    film: f32,
    normal: vec3<f32>,
}

fn wet_ripple_height(uv: vec2<f32>) -> f32 {
    let t = 0.10 * frame.time;
    let a = textureSampleLevel(wave_noise, wave_noise_sampler, uv + vec2(0.6, 0.8) * t, 0.0).g;
    let b = textureSampleLevel(wave_noise, wave_noise_sampler, uv + vec2(-0.385, -0.923) * t, 0.0).g;
    return 0.5 * (a + b);
}

// Procedural dual-harmonic concentric raindrop splash rings on puddles (primary impact
// wave + trailing capillary ripple, varied phase/speed per cell).
fn rain_splash_slope(world_xz: vec2<f32>, scale: f32, speed: f32, seed: vec2<f32>) -> vec2<f32> {
    let scaled = world_xz * scale + seed;
    let cell = floor(scaled);
    let local = fract(scaled) - 0.5;
    let h1 = fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
    let h2 = fract(sin(dot(cell, vec2(269.5, 183.3))) * 43758.5453);
    let phase = fract(frame.time * speed + h1 * 1.37);
    let center = (vec2(h1, h2) - 0.5) * 0.46;
    let delta = local - center;
    let dist = length(delta);
    let radius = phase * 0.47;
    let dr1 = (dist - radius) * 26.0;
    let dr2 = (dist - radius * 0.72) * 34.0;
    let om_phase = 1.0 - phase;
    let fade = om_phase * om_phase * smoothstep(0.0, 0.08, phase);
    let primary = sin((dist - radius) * 40.0) * exp(-dr1 * dr1);
    let capillary = 0.45 * sin((dist - radius * 0.72) * 62.0) * exp(-dr2 * dr2);
    return delta / max(dist, 0.02) * ((primary + capillary) * fade * 0.14);
}

fn apply_wet_surface(input: MaterialInput, normal: vec3<f32>, porosity: f32) -> WetSurface {
    if !wet_surfaces {
        return WetSurface(1.0, 0.0, normal);
    }

    let world = input.position + frame.camera_position;
    var darkening = 1.0;
    var film = 0.0;
    var wet_normal = normal;

    // Capillary-fringe wet shorelines along sea-level banks (y = 62..63.35) and emergent sandbars:
    // darkens porous sand/dirt/stone and adds a damp specular sheen without wetting dry inland y>=63 plains.
    if input.light.y > 0.65 && porosity > 0.20 && world.y >= 61.95 && world.y <= 63.45 {
        let wave_wobble = (textureSampleLevel(wave_noise, wave_noise_sampler, world.xz * 0.09, 0.0).g - 0.5) * 0.22;
        let bank_band = select(0.0, 1.0 - smoothstep(62.82, 63.35, world.y + wave_wobble), abs(input.normal.y) < 0.5);
        let bar_band = select(0.0, 1.0 - smoothstep(62.65, 62.98, world.y + wave_wobble * 0.5), input.normal.y > 0.9);
        let capillary = max(bank_band, bar_band) * smoothstep(0.65, 0.90, input.light.y);
        if capillary > 0.001 {
            darkening *= 1.0 - capillary * (0.18 + 0.28 * porosity);
            film = max(film, capillary * (0.28 + 0.22 * (1.0 - porosity)));
        }
    }

    if frame.wetness <= 0.0 || frame.temperature < 0.15
        || frame.downfall <= 0.0 || input.normal.y < 0.95 || input.light.y < 0.9 {
        return WetSurface(darkening, film, wet_normal);
    }

    // Two-octave puddle field for natural pooling and crisp wet rims.
    let broad = textureSampleLevel(wave_noise, wave_noise_sampler, world.xz * 0.022, 0.0).a;
    let detail = textureSampleLevel(wave_noise, wave_noise_sampler, world.xz * 0.085 + vec2(0.37, 0.61), 0.0).g;
    let puddle_field = 0.72 * broad + 0.28 * detail;

    let exposure = smoothstep(0.9, 1.0, input.light.y);
    let wet = frame.wetness * exposure;
    let wet_rim = smoothstep(0.32, 0.52, puddle_field) * wet;
    let meniscus = smoothstep(0.40, 0.47, puddle_field) * (1.0 - smoothstep(0.49, 0.56, puddle_field)) * wet;
    let pool = smoothstep(0.46, 0.58, puddle_field) * wet;
    let rain_film = pool * (1.0 - 0.68 * porosity);
    film = max(film, rain_film);
    // Capillary puddle-rim darkening mimics water wicked into porous stone/dirt at puddle edges.
    darkening *= 1.0 - wet * (0.07 + 0.14 * porosity) - wet_rim * (0.14 + 0.24 * porosity) - meniscus * 0.12 * porosity;

    wet_normal = normalize(mix(normal, input.normal, rain_film));
    if rain_film > 0.05 && frame.rain > 0.0 && dot(input.position, input.position) < 900.0 {
        let uv = world.xz * 0.30;
        let h = wet_ripple_height(uv);
        let wind_slope = vec2(wet_ripple_height(uv + vec2(0.03, 0.0)) - h,
            wet_ripple_height(uv + vec2(0.0, 0.03)) - h);
        let splash = rain_splash_slope(world.xz, 1.35, 1.30, vec2(0.0, 0.0))
            + rain_splash_slope(world.xz, 1.85, 1.62, vec2(17.3, 9.1))
            + rain_splash_slope(world.xz, 2.35, 1.12, vec2(-11.4, 23.7)) * 0.65;
        let ripple = normalize(vec3(wind_slope.x + splash.x, 1.0, wind_slope.y + splash.y));
        wet_normal = normalize(mix(wet_normal, ripple, 0.64 * rain_film * frame.rain));
    }
    return WetSurface(darkening, film, wet_normal);
}
