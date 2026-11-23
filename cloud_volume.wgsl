// Integrate scattering and transmittance, with an opacity-weighted distance for
// layer ordering and wind-aware cloud history.
struct CloudResult {
    color: vec3<f32>,
    transmittance: f32,
    distance: f32,
    layer: u32,
}

const CLEAR_CLOUD = CloudResult(vec3(0.0), 1.0, 0.0, 0u);

fn cloud_slab(direction: vec3<f32>, distance: f32, base: f32, top: f32) -> vec2<f32> {
    let y = frame.camera_position.y;
    if abs(direction.y) < 0.0001 {
        return vec2(0.0, select(0.0, distance, y > base && y < top));
    }
    let a = (base - y) / direction.y;
    let b = (top - y) / direction.y;
    return vec2(max(0.0, min(a, b)), min(distance, max(a, b)));
}

fn cloud_sphere(ray: vec3<f32>, altitude: f32) -> vec2<f32> {
    let eye = frame.camera_position.y;
    let b = (CLOUD_PLANET_RADIUS + eye - 63.0) * ray.y;
    let c = (eye - altitude) * (2.0 * CLOUD_PLANET_RADIUS + eye + altitude - 126.0);
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return vec2(-1.0);
    }
    let root = sqrt(discriminant);
    let far_root = -b - select(-root, root, b >= 0.0);
    if abs(far_root) < 1e-5 {
        return vec2(-b);
    }
    let near_root = c / far_root;
    return vec2(min(far_root, near_root), max(far_root, near_root));
}

fn cloud_shell(ray: vec3<f32>, distance: f32, bounds: vec2<f32>) -> vec2<f32> {
    let outer = cloud_sphere(ray, bounds.y);
    let inner = cloud_sphere(ray, bounds.x);
    let eye = frame.camera_position.y;
    var start = max(0.0, outer.x);
    var end = min(distance, outer.y);
    if eye < bounds.x {
        start = max(start, inner.y);
        // A ray through the planet cannot reach the cloud layer on the far side.
        if cloud_sphere(ray, min(63.0, eye - 1.0)).y > 0.0 {
            return vec2(0.0);
        }
    } else if inner.x >= 0.0 {
        end = min(end, inner.x);
    }
    return vec2(start, end);
}

struct CloudLayerSettings {
    bounds: vec2<f32>,
    kind: u32,
    stratus: f32,
    coverage: vec2<f32>,
    height_scale: f32,
    extinction: f32,
    upper_shadow: f32,
    congestus: f32,
}

fn volume_density(relative: vec3<f32>, layer: CloudLayerSettings) -> f32 {
    let world = frame.camera_position + relative;
    if blocky_clouds {
        return blocky_density(world, layer.kind);
    }
    let height = cloud_radial_height(world.y, relative.xz);
    if height <= layer.bounds.x || height >= layer.bounds.y {
        return 0.0;
    }
    switch layer.kind {
        case 0u: {
            return cumulus_density(world, height, cloud_coverage(world.xz), layer.stratus, layer.height_scale);
        }
        case 1u: {
            return congestus_density(world, relative, height, layer.congestus);
        }
        default: {
            return mid_cloud_density(world, height, layer.stratus, layer.coverage);
        }
    }
}

fn volume_density_fast(relative: vec3<f32>, layer: CloudLayerSettings) -> f32 {
    let world = frame.camera_position + relative;
    if blocky_clouds {
        return blocky_density(world, layer.kind);
    }
    let height = cloud_radial_height(world.y, relative.xz);
    if height <= layer.bounds.x || height >= layer.bounds.y {
        return 0.0;
    }
    switch layer.kind {
        case 0u: {
            return cumulus_density_fast(height, cloud_coverage(world.xz), layer.stratus, layer.height_scale);
        }
        case 1u: {
            return congestus_density_fast(world, relative, height, layer.congestus);
        }
        default: {
            return mid_cloud_density_fast(world, height, layer.stratus, layer.coverage);
        }
    }
}

fn layer_optical_depth(origin: vec3<f32>, direction: vec3<f32>, thickness: f32,
    layer: CloudLayerSettings, count: u32, jitter: f32) -> f32 {
    var step_length = 0.1 * thickness / f32(count);
    var position = origin;
    var depth = 0.0;
    for (var i = 0u; i < count; i += 1u) {
        step_length *= 2.0;
        depth += volume_density_fast(position + step_length * jitter * direction, layer) * step_length;
        position += step_length * direction;
    }
    return depth;
}

fn draine_phase(cosine: f32, g: f32, alpha: f32) -> f32 {
    let g2 = g * g;
    let d = max(1.0 + g2 - 2.0 * g * cosine, 1e-4);
    let hg = (1.0 - g2) / (4.0 * PHOTON_PI * d * sqrt(d));
    return hg * (1.0 + alpha * cosine * cosine) / (1.0 + alpha * (1.0 + 2.0 * g2) * (1.0 / 3.0));
}

fn cloud_phase_single(cosine: f32) -> f32 {
    // Jendersie & d'Eon (SIGGRAPH 2023) analytic Draine + HG Mie fit for cloud
    // water droplets (r ~ 8-10 um): sharp forward silver lining, subtle anti-solar
    // backscatter glory (cosine -> -1), and delicate 38-deg cloudbow arc (cosine ~ -0.788).
    let forward_hg = photon_phase(cosine, 0.88);
    let draine = draine_phase(cosine, 0.66, 18.0);
    let back_glory = photon_phase(cosine, -0.22);
    let glory_peak = 0.08 * exp(-18.0 * (1.0 + cosine));
    let bow_d = cosine + 0.788;
    let cloudbow = 0.045 * exp(-34.0 * bow_d * bow_d);
    return 0.56 * forward_hg + 0.30 * draine + 0.14 * back_glory + glory_peak + cloudbow;
}

// Curved-Earth altitude solar coloring: higher cloud decks (mid altocumulus, congestus
// anvils, cirrus) stay bathed in warm golden-peach sunlight even after the ground and
// lower cloud bases fall into sunset/twilight Earth shadow.
fn cloud_altitude_sun(base_sun: vec3<f32>, altitude_m: f32) -> vec3<f32> {
    let sun_y = frame.sun_direction.y;
    let twilight_window = smoothstep(-0.11, -0.02, sun_y) * (1.0 - smoothstep(0.05, 0.18, sun_y))
        * (1.0 - 0.80 * frame.rain);
    if twilight_window <= 0.001 {
        return base_sun;
    }
    // Scale Minecraft cloud altitude into atmospheric km (183m -> 1.8km, 700m -> 8km).
    let alt_km = clamp((altitude_m - 63.0) * 0.011, 0.0, 12.0);
    let elev_lift = sqrt(2.0 * alt_km / 6371.0);
    let high_lit = smoothstep(-0.075, 0.025, sun_y + elev_lift * 1.45);
    let alt_factor = clamp((alt_km - 1.2) * 0.14, 0.0, 1.0);
    // Warm golden-peach high-altitude alpenglow vs cooler magenta-violet lower bases.
    let alpenglow = vec3(1.42, 0.76, 0.44) * (high_lit * alt_factor * twilight_window * 0.55);
    let base_cool = mix(vec3(1.0), vec3(0.84, 0.76, 0.96), (1.0 - alt_factor) * twilight_window * 0.25);
    let luma = max(dot(base_sun, frame.luminance), 0.06);
    return base_sun * base_cool + alpenglow * (0.28 + 0.72 * luma);
}

fn cloud_phase_multi(cosine: f32, g: vec3<f32>) -> f32 {
    return 0.65 * photon_phase(cosine, g.x) + 0.10 * photon_phase(cosine, g.y)
        + 0.25 * photon_phase(cosine, -g.z);
}

override neural_clouds: bool = true;

// Zero-spill mat4x4/mat4x2 Neural Cloud Multi-Scattering MLP v2 (8 -> 16 -> 16 -> 2, ReLU).
// Trained on 36,000 Monte Carlo cumulus states (9.2M rays, 16 bounces, SIGGRAPH 2023 Draine+HG phase).
// Validation R2_sun=0.9542 (E=0.9993), R2_sky=0.8335 (E=0.9859).
fn neural_cloud_multi_scatter(tau_sun: f32, tau_sky: f32, tau_gnd: f32,
    cosine: f32, density: f32, height: f32) -> vec2<f32> {
    let x0 = vec4(exp(-0.5 * tau_sun), 1.0 / (1.0 + 0.25 * tau_sun), exp(-0.5 * tau_sky), 1.0 / (1.0 + 0.35 * tau_sky));
    let x1 = vec4(exp(-0.4 * tau_gnd), cosine, 1.0 - exp(-2.5 * density), height);
    let h1_0 = max(mat4x4<f32>(
        vec4(0.57227, -2.08446, 0.27548, -0.26222),
        vec4(-0.52446, -0.45643, -0.04326, 0.08643),
        vec4(1.37316, -0.24428, -0.01467, -0.41984),
        vec4(-0.35982, 0.91110, 0.48895, 0.66747)
    ) * x0 + mat4x4<f32>(
        vec4(0.51095, -0.48617, -1.21834, -0.03262),
        vec4(0.00638, 0.17631, 0.01877, 0.35494),
        vec4(0.14659, -0.39692, 0.30571, 0.47375),
        vec4(0.08128, 0.04524, 0.04075, 0.00448)
    ) * x1 + vec4(0.15817, 0.23826, 0.78374, -0.24283), vec4(0.0));
    let h1_1 = max(mat4x4<f32>(
        vec4(-0.32149, 1.18969, -0.52039, -0.03122),
        vec4(0.55676, 0.49874, 0.58250, 0.09141),
        vec4(0.11736, -1.03165, -0.22262, -0.12745),
        vec4(0.03149, -0.99597, -0.50629, -0.10148)
    ) * x0 + mat4x4<f32>(
        vec4(-1.10970, 0.04311, -0.25518, -0.10891),
        vec4(0.20385, 0.02277, 0.06076, 0.15861),
        vec4(-0.23933, -0.00241, 0.47514, 0.07197),
        vec4(-0.00130, 0.01172, -0.01228, 0.58114)
    ) * x1 + vec4(0.70506, 0.27946, 0.48070, 0.14622), vec4(0.0));
    let h1_2 = max(mat4x4<f32>(
        vec4(-0.59853, -0.03155, 0.43736, 1.10265),
        vec4(0.79638, 0.10860, 0.23052, 0.46771),
        vec4(0.52934, -0.46356, -0.41418, -1.16169),
        vec4(0.15872, -0.48898, -0.13613, -0.41379)
    ) * x0 + mat4x4<f32>(
        vec4(-0.34239, 0.43657, -0.18493, -0.50686),
        vec4(0.21443, 0.95464, 0.45212, 0.15272),
        vec4(0.15970, 0.03955, 0.26011, -0.03509),
        vec4(0.03612, -0.05691, -0.00591, 0.00570)
    ) * x1 + vec4(0.08976, -0.13769, 0.01192, 0.45535), vec4(0.0));
    let h1_3 = max(mat4x4<f32>(
        vec4(-0.36066, 0.46449, -0.32371, -0.04506),
        vec4(-0.08921, -1.82715, -1.03471, 0.10842),
        vec4(0.34839, 0.03662, -0.02761, 0.56402),
        vec4(0.44629, 0.25336, 0.22340, -0.61136)
    ) * x0 + mat4x4<f32>(
        vec4(-0.39116, -0.07257, -0.09634, -0.10115),
        vec4(1.61579, -0.04184, 0.65616, -0.10357),
        vec4(0.05940, 0.09014, -0.04726, -1.06705),
        vec4(0.00591, 0.01123, -0.00234, -0.42396)
    ) * x1 + vec4(-0.69061, 0.93950, 0.50117, -0.05417), vec4(0.0));
    let h2_0 = max(mat4x4<f32>(
        vec4(-0.42182, 0.32636, -0.33644, 0.57049),
        vec4(0.28658, -0.34883, -0.08567, -0.02167),
        vec4(0.58441, -0.17810, -0.29286, -0.21052),
        vec4(0.14770, 0.04067, 0.48843, 0.49544)
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.19059, -0.49788, 0.55045, 0.58021),
        vec4(0.95522, -1.83733, 0.00831, 0.17647),
        vec4(0.21316, 0.06330, -0.21854, -0.31190),
        vec4(0.00221, -0.02672, 0.09782, -0.07995)
    ) * h1_1 + mat4x4<f32>(
        vec4(0.62289, 0.05844, -1.23028, 0.17997),
        vec4(0.78359, -0.05320, -0.15370, 0.25283),
        vec4(-0.19716, 0.19078, 0.37417, -0.01524),
        vec4(-0.71237, 1.85578, -0.71966, -0.31256)
    ) * h1_2 + mat4x4<f32>(
        vec4(0.08292, -0.03241, 0.43527, -0.09698),
        vec4(-1.19287, -0.10614, -0.06410, 0.11211),
        vec4(-0.35716, 0.18680, 0.00445, -0.32552),
        vec4(0.13470, 0.00320, -0.36056, -0.16931)
    ) * h1_3 + vec4(-0.34284, -0.53214, 0.65228, -0.78563), vec4(0.0));
    let h2_1 = max(mat4x4<f32>(
        vec4(0.11397, -0.20170, 0.11766, -0.16331),
        vec4(-0.32410, -1.08460, 0.21519, 0.40550),
        vec4(0.25365, -0.26840, 0.13260, 0.33571),
        vec4(-0.20829, -0.31981, 0.35957, 0.11506)
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.36861, -0.02152, -0.10246, -0.26815),
        vec4(-0.49769, -0.08367, -0.64328, 0.91913),
        vec4(0.30143, -0.09405, -0.01327, 0.42864),
        vec4(-0.01882, -0.37735, -0.05552, -0.01649)
    ) * h1_1 + mat4x4<f32>(
        vec4(-0.38215, 0.21924, 0.33688, 0.38487),
        vec4(-0.08333, 0.24330, 0.12003, 0.18289),
        vec4(0.36818, -0.54473, 0.21057, 0.40881),
        vec4(0.79510, -0.01174, 0.38041, -0.61594)
    ) * h1_2 + mat4x4<f32>(
        vec4(-0.51946, 0.44564, -0.10835, 0.17518),
        vec4(0.19940, -0.28635, 0.16164, -1.29168),
        vec4(-0.04962, -0.44122, -0.20669, -0.54542),
        vec4(-0.48750, -0.73686, -0.50887, -0.04162)
    ) * h1_3 + vec4(-0.00027, -0.08520, -0.19018, 0.18307), vec4(0.0));
    let h2_2 = max(mat4x4<f32>(
        vec4(0.44291, -1.17692, -0.26401, 0.12512),
        vec4(-1.15027, 0.29139, -0.37281, -0.62005),
        vec4(-0.24612, -0.31729, -0.28864, -0.06489),
        vec4(0.23731, -0.00134, 0.49723, 0.01094)
    ) * h1_0 + mat4x4<f32>(
        vec4(0.58799, -0.89047, 0.18286, 0.79861),
        vec4(1.30529, 0.03859, 0.59830, 0.03370),
        vec4(-0.03103, 0.41373, -0.29155, 0.56308),
        vec4(-0.06617, 0.15515, 0.10015, -0.01407)
    ) * h1_1 + mat4x4<f32>(
        vec4(0.23011, 0.81560, -0.26804, -0.67367),
        vec4(-0.16067, -0.44309, 0.32385, 0.06586),
        vec4(-0.33850, 0.53269, 0.30547, 0.20699),
        vec4(-1.15587, 0.17100, -0.50174, 0.47379)
    ) * h1_2 + mat4x4<f32>(
        vec4(-0.04174, -0.37977, 1.24238, 0.10282),
        vec4(-0.62978, -0.47813, -0.16963, 0.01248),
        vec4(-0.30702, 0.33179, -0.76784, -0.04306),
        vec4(0.34862, 0.06988, 0.77611, -0.09859)
    ) * h1_3 + vec4(-0.57461, 0.31306, -0.54204, 0.02920), vec4(0.0));
    let h2_3 = max(mat4x4<f32>(
        vec4(-0.13316, -1.11174, 0.04443, -0.24993),
        vec4(0.14791, -0.11469, -0.44622, -0.24277),
        vec4(0.14713, 0.42621, 0.62158, -0.12411),
        vec4(-0.06856, 0.01573, 0.44477, -0.60592)
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.08393, -0.53605, 0.05134, -0.63105),
        vec4(-0.12459, 0.32068, 0.02661, -0.80481),
        vec4(-0.06004, 0.84419, 0.04679, 0.06080),
        vec4(-0.81047, 0.17582, -0.05462, 0.24129)
    ) * h1_1 + mat4x4<f32>(
        vec4(0.05263, -0.27377, 0.35845, -0.01146),
        vec4(0.25379, 0.35537, 0.26242, 0.34003),
        vec4(-0.47836, 0.43666, 0.30798, -0.28244),
        vec4(-0.31019, -0.22357, -0.19277, 0.46717)
    ) * h1_2 + mat4x4<f32>(
        vec4(-0.26417, -0.43660, -0.01402, 0.49381),
        vec4(-0.58787, 0.59996, 0.27151, -0.40340),
        vec4(0.25614, -0.08134, -0.21257, -0.24183),
        vec4(-0.39340, 0.39452, -0.09935, -0.19213)
    ) * h1_3 + vec4(-0.10536, -0.36039, -0.26957, -0.12328), vec4(0.0));
    let out_log = mat4x2<f32>(
        vec2(0.56312, -0.01937),
        vec2(-1.23236, -0.64265),
        vec2(-0.12535, -0.07007),
        vec2(0.11384, -0.63714)
    ) * h2_0 + mat4x2<f32>(
        vec2(-0.05318, 0.26058),
        vec2(-0.44909, 0.41128),
        vec2(-0.20509, 0.11497),
        vec2(0.39358, 0.06773)
    ) * h2_1 + mat4x2<f32>(
        vec2(-0.43729, -0.03774),
        vec2(0.49476, -0.09592),
        vec2(1.06267, 0.01292),
        vec2(0.06536, -0.23232)
    ) * h2_2 + mat4x2<f32>(
        vec2(-0.05022, -0.09521),
        vec2(-0.12037, -0.35265),
        vec2(0.07416, 0.34146),
        vec2(-0.44618, 0.25974)
    ) * h2_3 + vec2(0.03544, 0.06469);
    return max(exp(max(out_log, vec2(0.0))) - vec2(1.0), vec2(0.0)) * (1.0 / 6.0);
}

fn cloud_scattering(density: f32, depths: vec3<f32>, transmission: f32, cosine: f32,
    phase0: f32, bounced_light: f32, height: f32, storm_weight: f32, powder_cos: f32,
    falloff: f32, layer: CloudLayerSettings) -> vec2<f32> {
    let extinction = layer.extinction;
    let albedo_s = mix(1.0, 0.66, frame.rain);
    let base_ao = mix(mix(0.76, 0.34, storm_weight), 1.08, smoothstep(0.0, 0.72, height));

    if neural_clouds {
        // Exact analytic single scattering (preserves razor-sharp Draine + HG solar silver linings)
        // + path-traced neural MLP infinite-bounce multiple scattering.
        let tau = extinction * depths;
        let vis1 = exp(-tau);
        let single_sun = vis1.x * phase0 * (1.0 - 0.5 * layer.upper_shadow)
            + vis1.z * PHOTON_ISOTROPIC * bounced_light
            + 0.5 * vis1.y * PHOTON_ISOTROPIC * layer.upper_shadow;
        let single_sky = vis1.y * PHOTON_ISOTROPIC * base_ao;
        let multi = neural_cloud_multi_scatter(tau.x, tau.y, tau.z, cosine, density, height);
        let multi_sun = multi.x * (1.0 - 0.45 * layer.upper_shadow) + vis1.z * 0.35 * PHOTON_ISOTROPIC * bounced_light;
        let multi_sky = multi.y * base_ao;
        let scatter_s1 = extinction * albedo_s;
        let scatter_ms = extinction * albedo_s * albedo_s;
        let light = vec2(scatter_s1 * single_sun + scatter_ms * multi_sun,
            scatter_s1 * single_sky + scatter_ms * multi_sky);
        return light * ((1.0 - transmission) / extinction);
    }

    var scatter = extinction * albedo_s;
    var extinct = extinction;
    var phase = phase0;
    const LOG2_CLOUD_G = vec3(-0.7369656, -0.1520031, -1.7369656);
    var g = exp2((1.0 + depths.x * 10.0) * LOG2_CLOUD_G);
    let powder_density = density * (1.0 + layer.stratus);
    var powder = mix(PHOTON_PI * powder_density / (powder_density + 0.15), 1.0, powder_cos);
    var light = vec2(0.0);
    // 3 energy-compensated multi-scattering orders fallback when neural_clouds is disabled.
    for (var order = 0u; order < 3u; order += 1u) {
        let visibility = exp(-extinct * depths);
        let energy_scale = select(1.0, 1.28, order == 2u);
        let s = scatter * energy_scale;
        light.x += s * (visibility.x * phase * (1.0 - 0.5 * layer.upper_shadow)
            + visibility.z * PHOTON_ISOTROPIC * bounced_light
            + 0.5 * visibility.y * PHOTON_ISOTROPIC * layer.upper_shadow);
        light.y += s * visibility.y * PHOTON_ISOTROPIC * base_ao;
        if order < 2u {
            scatter *= falloff * powder;
            extinct *= 0.4;
            g *= 0.8;
            powder = mix(powder, sqrt(powder), 0.5);
            phase = cloud_phase_multi(cosine, g);
        }
    }
    return light * ((1.0 - transmission) / extinction);
}

fn cloud_aerial(color: vec3<f32>, transmittance: f32, ray: vec3<f32>, distance: f32, haze: vec3<f32>) -> vec3<f32> {
    let scale = mix(1.4, 0.45, smoothstep(0.0, 0.45, abs(ray.y)));
    var air = exp(-distance * scale * mix(vec3(0.00014, 0.00024, 0.00040), vec3(0.00043), frame.rain));
    air = mix(air, vec3(air.x), 0.8 * frame.rain);
    let rain_sky = mix(haze, photon_ambient_irradiance() / TAU, frame.rain * 0.95);
    return mix(rain_sky * (1.0 - transmittance), color, air);
}

fn cloud_layer_march(ray: vec3<f32>, limit: f32, steps_top: u32, jitter: f32,
    layer: CloudLayerSettings, sun: vec3<f32>, ambient: vec3<f32>, haze: vec3<f32>,
    terrain_clipped: bool) -> CloudResult {
    let interval_limit = select(limit, max(limit, 12000.0), terrain_clipped);
    var interval = cloud_shell(ray, interval_limit, layer.bounds);
    if blocky_clouds {
        interval = cloud_slab(ray, interval_limit, layer.bounds.x, layer.bounds.y);
    }
    let start = max(interval.x, select(0.0, 3000.0 / max(length(ray.xz), 0.001), layer.kind == 1u && !blocky_clouds));
    let travel = min(2000.0, min(limit, interval.y) - start);
    if travel <= 0.0 {
        return CLEAR_CLOUD;
    }
    let sample_count = mix(2.0 * f32(steps_top), f32(steps_top), abs(ray.y));
    let steps = u32(ceil(sample_count));
    // Keep the sample grid independent of terrain depth. Otherwise moving the
    // clipping surface shifts every sample in the cloud, causing popping.
    var grid_travel = travel;
    if terrain_clipped {
        grid_travel = min(2000.0, max(0.0, interval.y - start));
    }
    let step_size = grid_travel / sample_count;
    let thickness = layer.bounds.y - layer.bounds.x;
    let inv_thickness = 1.0 / thickness;
    let light = frame.light_direction;
    let cosine = clamp(dot(ray, light), -1.0, 1.0);
    let phase0 = cloud_phase_single(cosine);
    let bounced = 0.4 * max(0.0, light.y) / PHOTON_PI;
    let storm_weight = clamp(layer.stratus * 0.55 + frame.rain * 0.65 + frame.thunder * 0.35, 0.0, 1.0);
    let half_cos = 0.5 * cosine + 0.5;
    let powder_cos = 0.8 * cloud_square(half_cos);
    let init_scatter = layer.extinction * mix(1.0, 0.66, frame.rain);
    let falloff = 0.55 * mix(cloud_lift(clamp(init_scatter, 0.0, 1.0), 0.33), 1.0, half_cos);
    var result = CloudResult(vec3(0.0), 1.0, 0.0, layer.kind);
    var weight_sum = 0.0;
    var energy = vec2(0.0);
    const MIN_TRANSMITTANCE: f32 = 0.075;
    for (var i = 0u; i < steps && result.transmittance > MIN_TRANSMITTANCE; i += 1u) {
        let remaining = travel - f32(i) * step_size;
        if remaining <= 0.0 {
            break;
        }
        let segment = min(step_size, remaining);
        let along = start + f32(i) * step_size + jitter * segment;
        let point = ray * along;
        // Fast 2D macro envelope gate skips 3D Worley detail & secondary light marches
        // in clear sky while staying 100% monotonic (zero back-step ping-pong or hollow rings).
        if !blocky_clouds && volume_density_fast(point, layer) <= 0.0005 {
            continue;
        }
        let density = volume_density(point, layer) * (1.0 - smoothstep(0.95, 1.0, (along - start) * 0.0005));
        if density > 0.001 {
            let transmission = exp(-density * segment * layer.extinction);
            let light_dither = fract(jitter + f32(i) * 0.381966);
            let sky_dither = fract(jitter + f32(i) * 0.7548777);
            let light_depth = layer_optical_depth(point, light, thickness, layer, select(3u, 2u, blocky_clouds), light_dither);
            let sky_depth = layer_optical_depth(point, vec3(0.0, 1.0, 0.0), thickness, layer, 1u, sky_dither);
            let height = clamp((cloud_radial_height(frame.camera_position.y + point.y, point.xz)
                - layer.bounds.x) * inv_thickness, 0.0, 1.0);
            let ground_depth = mix(density, 1.0, max(0.0, height * 2.0 - 1.0)) * height * thickness;
            energy += result.transmittance * cloud_scattering(density, vec3(light_depth, sky_depth, ground_depth),
                transmission, cosine, phase0, bounced, height, storm_weight, powder_cos, falloff, layer);
            let visible_opacity = result.transmittance * (1.0 - transmission);
            result.distance += along * visible_opacity;
            weight_sum += visible_opacity;
            result.transmittance *= transmission;
        }
    }
    if weight_sum <= 0.0 {
        return CLEAR_CLOUD;
    }
    // Upstream Photon remaps min_transmittance (0.075) to 0.0.
    result.transmittance = cloud_linear(MIN_TRANSMITTANCE, 1.0, result.transmittance);
    result.distance /= weight_sum;
    // RDR2 warm sunset/sunrise cloud under-lighting (locked to sun_direction so it
    // never flips to the moon after sunset). Sheet lightning is applied in sky.wgsl
    // at full framerate so thunderstorms never trigger cloud-history resets or stutters.
    let sun_y = frame.sun_direction.y;
    let sun_cos = clamp(dot(ray, frame.sun_direction), -1.0, 1.0);
    let sunset_band = smoothstep(-0.09, 0.04, sun_y) * (1.0 - smoothstep(0.06, 0.24, sun_y))
        * (1.0 - 0.85 * frame.rain) * (1.0 - 0.55 * layer.stratus);
    let under_lit = vec3(1.18, 0.56, 0.34) * (sunset_band * (0.20 + 0.45 * pow(max(sun_cos * 0.5 + 0.5, 0.0), 2.0))
        * (1.0 - result.transmittance) * 0.28);
    result.color = cloud_aerial(sun * energy.x + ambient * energy.y + under_lit,
        result.transmittance, ray, result.distance, haze);
    return result;
}

fn cloud_curl(coord: vec2<f32>) -> vec2<f32> {
    let broad = cloud_noise(coord).yz;
    let medium = cloud_noise(2.0 * coord).yz;
    let fine = cloud_noise(4.0 * coord).yz;
    let gradient = 0.5 * broad + 0.25 * medium + 0.125 * fine;
    return vec2(gradient.y, -gradient.x);
}

fn cirrus_density(world: vec2<f32>, cirrus: f32, cirrocumulus: f32) -> f32 {
    let coord = world + cloud_drift(frame.time, 3u);
    let curl = cloud_curl(0.0002 * coord);
    let broad = 0.7 * photon_cloud_noise(coord * 0.00001 + curl * 0.004).r
        + 0.3 * photon_cloud_noise(coord * 0.00008 + curl * 0.008).r;
    var sheet = cloud_linear(0.7 - cirrus, 1.0, broad);
    var frequency = 0.0002;
    var strength = 0.1;
    var amplitude = 0.2;
    for (var octave = 0u; octave < 4u; octave += 1u) {
        let drift = f32(octave) * 0.3 * cloud_drift(frame.time, 3u);
        sheet -= amplitude * photon_cloud_noise((coord + drift) * frequency + curl * strength).r;
        frequency *= 2.0;
        strength *= 4.0;
        amplitude *= 0.6;
    }
    let cover = 5.0 * cloud_linear(0.25, 0.9, cirrocumulus * photon_cloud_noise(coord * 0.000026 + 0.25).a);
    var puffs = cloud_linear(1.0 - max(0.001, cover), 1.0,
        photon_dampen(photon_cloud_noise(coord * 0.00025 + 0.033 * curl).a));
    let curl_puffs = cloud_curl(coord * 0.01);
    puffs -= cloud_square(photon_cloud_noise(coord * 0.0005 + 0.003 * curl_puffs).g);
    puffs -= 0.5 * cloud_square(photon_cloud_noise(coord * 0.002 + 0.007 * curl_puffs).g);
    puffs -= 0.1 * cloud_square(photon_cloud_noise(coord * 0.008 + 0.03 * curl_puffs).g);
    return pow(max(0.0, sheet), 3.0) + 0.25 * pow(max(0.0, puffs), 4.0);
}

fn cloud_sheet(ray: vec3<f32>, distance: f32, ambient: vec3<f32>, sun: vec3<f32>, haze: vec3<f32>) -> CloudResult {
    let interval = cloud_shell(ray, max(distance, 12000.0), vec2(588.0, 738.0));
    let visible = min(distance, interval.y) - interval.x;
    if visible <= 0.0 {
        return CLEAR_CLOUD;
    }
    // Anchor the texture to the sheet while terrain progressively clips its optical depth.
    let along = 0.5 * (interval.x + interval.y);
    let world = frame.camera_position + ray * along;
    let humidity = cloud_weather().y;
    let temperature = cloud_weather().x;
    let cirrus = cloud_cover / 0.55 * (0.5 * (0.6 - 0.4 * cloud_square(cloud_linear(0.5, 0.9, temperature))
        + 0.4 * (1.0 - cloud_linear(0.0, 0.2, temperature))) * (1.0 - 0.5 * cloud_linear(0.5, 0.75, humidity))
        + 0.5 * frame.rain);
    let patches = 0.8 * cloud_cover / 0.55 * photon_dampen((1.0 - 0.3 * cloud_linear(0.5, 1.0, temperature))
        * (0.5 + 0.5 * cloud_linear(0.4, 1.0, humidity)) * pow(weather_wind(), 1.5));
    let density = cirrus_density(world.xz, cirrus, patches);
    if density <= 0.0005 {
        return CLEAR_CLOUD;
    }
    let light_shadow = cirrus_density(world.xz + frame.light_direction.xz * 65.0, cirrus, patches);
    let depth = min(360.0, interval.y - interval.x) * (visible / (interval.y - interval.x)) * density;
    let transmission = exp(-0.38 * depth);
    let phase = cloud_phase_single(clamp(dot(ray, frame.light_direction), -1.0, 1.0));
    let color = (ambient * (0.09 + 0.10 * exp(-2.0 * density))
        + sun * (phase * 0.55 * exp(-5.0 * light_shadow) + 0.04)) * (1.0 - transmission);
    return CloudResult(cloud_aerial(color, transmission, ray, along, haze), transmission, along, 3u);
}

fn noctilucent_sheet(ray: vec3<f32>, distance: f32) -> CloudResult {
    let twilight = smoothstep(0.025, 0.11, -frame.sun_direction.y)
        * (1.0 - smoothstep(0.14, 0.32, -frame.sun_direction.y)) * (1.0 - frame.rain);
    if twilight < 0.001 || ray.y <= 0.01 || (frame.moon_phase % 3u) == 0u {
        return CLEAR_CLOUD;
    }
    let along = (775.0 - frame.camera_position.y) / ray.y;
    if along <= 0.0 || along >= distance {
        return CLEAR_CLOUD;
    }
    let coord = 0.25 * (frame.camera_position + ray * along).xz;
    let curl = cloud_curl(0.0002 * coord);
    var density = photon_dampen(photon_cloud_noise(0.00001 * coord + 0.02 * curl).g);
    var amplitude = 0.3;
    var frequency = 0.0001;
    var strength = 0.05;
    for (var octave = 0u; octave < 4u; octave += 1u) {
        density += amplitude * photon_cloud_noise(coord * frequency + curl * strength).g;
        amplitude *= 0.5;
        frequency *= 1.5;
        strength *= 4.0;
    }
    let highlight = photon_dampen(photon_cloud_noise(0.00004 * coord + 0.02 * curl).g)
        - 0.15 * (1.0 - photon_cloud_noise(0.00014 * coord + 0.05 * curl).g);
    density = cloud_square(density + 2.0 * pow(max(0.0, highlight), 8.0));
    let opacity = density * twilight * 0.18 * cloud_cover;
    return CloudResult(vec3(0.16, 0.32, 0.68) * opacity, exp(-opacity * 0.35), along, 4u);
}

fn blend_clouds(a: CloudResult, b: CloudResult) -> CloudResult {
    if a.transmittance >= 0.99999 { return b; }
    if b.transmittance >= 0.99999 { return a; }
    var near = a;
    var far = b;
    if b.distance < a.distance {
        near = b;
        far = a;
    }
    return CloudResult(near.color + near.transmittance * far.color,
        near.transmittance * far.transmittance, near.distance, near.layer);
}

fn cloud_layers_impl(ray: vec3<f32>, distance: f32, steps: u32, jitter: f32, terrain_clipped: bool) -> CloudResult {
    if !volumetric_clouds || cloud_cover <= 0.0 {
        return CLEAR_CLOUD;
    }
    let sun = direct_atmosphere_light();
    let ambient = photon_ambient_irradiance() + storm_flash() * vec3(0.9, 1.6, 3.0);
    let haze = atmosphere(ray);
    if blocky_clouds {
        let first = CloudLayerSettings(vec2(CLOUD_LAYER_HEIGHT - 8.0, CLOUD_LAYER_HEIGHT + 8.0),
            0u, 0.0, vec2(0.0), 1.0, 0.19, 0.0, 0.0);
        let base = cloud_layer_march(ray, distance, steps, jitter, first, sun, ambient, haze, terrain_clipped);
        if second_blocky_layer {
            let second = CloudLayerSettings(vec2(BLOCKY_SECOND_HEIGHT - 8.0, BLOCKY_SECOND_HEIGHT + 8.0),
                1u, 0.0, vec2(0.0), 1.0, 0.19, 0.0, 0.0);
            return blend_clouds(base, cloud_layer_march(ray, distance, steps, jitter, second, sun, ambient, haze, terrain_clipped));
        }
        return base;
    }

    let stratus = cumulus_stratus();
    let mid_blend = mid_stratus();
    let mid_cover = mid_coverage(mid_blend);
    let upper_shadow = cloud_linear(0.7, 1.0, dot(mid_cover, vec2(0.25, 0.75)) * (1.0 + mid_blend))
        * photon_dampen(smoothstep(-0.05, 0.08, frame.sun_direction.y));
    let cover = cumulus_coverage_range(stratus);
    let extinction = mix(0.5, 1.0, smoothstep(0.0, 0.3, abs(frame.sun_direction.y)))
        * (1.0 - 0.33 * frame.rain) * (1.0 - 0.6 * upper_shadow)
        * (1.0 - 0.4 * cloud_linear(0.8, 1.0, cover.y * (1.0 - stratus))) * (1.0 - 0.5 * stratus);
    let low_sun = cloud_altitude_sun(sun, 0.5 * (CLOUD_BASE + CLOUD_TOP));
    let low = CloudLayerSettings(vec2(CLOUD_BASE, CLOUD_TOP), 0u, stratus, cover,
        cumulus_height_scale(stratus), extinction, upper_shadow, 0.0);
    var result = cloud_layer_march(ray, distance, steps, jitter, low, low_sun, ambient, haze, terrain_clipped);
    if result.transmittance <= 0.001 && frame.camera_position.y < CLOUD_BASE && result.distance < 3000.0 {
        return result;
    }

    if mid_cover.y > 0.001 {
        let mid_sun = cloud_altitude_sun(sun, 0.5 * (MID_CLOUD_BASE + MID_CLOUD_TOP));
        let middle = CloudLayerSettings(vec2(MID_CLOUD_BASE, MID_CLOUD_TOP), 2u, mid_blend,
            mid_cover, 1.0, 0.60, 0.0, 0.0);
        result = blend_clouds(result, cloud_layer_march(ray, distance, max(7u, steps / 2u),
            jitter, middle, mid_sun, ambient, haze, terrain_clipped));
    }
    let amount = congestus_amount();
    if amount > 0.001 && distance > 3000.0 && result.transmittance > 0.02 {
        let tower_sun = cloud_altitude_sun(sun, 0.5 * (CONGESTUS_BASE + CONGESTUS_TOP));
        let tower = CloudLayerSettings(vec2(CONGESTUS_BASE, CONGESTUS_TOP), 1u, 0.0, vec2(0.0),
            1.0, 0.8, 0.0, amount);
        let distant = cloud_layer_march(ray, distance, max(10u, steps / 2u), jitter, tower, tower_sun, ambient, haze, terrain_clipped);
        let fade = mix(1.0, distant.transmittance, cloud_linear(2250.0, 3000.0, result.distance));
        result.color *= fade;
        result.transmittance = mix(1.0, result.transmittance, fade);
        result = blend_clouds(result, distant);
    }
    if result.transmittance <= 0.001 && frame.camera_position.y < MID_CLOUD_BASE {
        return result;
    }
    let cirrus_sun = cloud_altitude_sun(sun, 660.0);
    return blend_clouds(blend_clouds(result, cloud_sheet(ray, distance, ambient, cirrus_sun, haze)),
        noctilucent_sheet(ray, distance));
}

fn cloud_layers(ray: vec3<f32>, distance: f32, steps: u32, jitter: f32) -> CloudResult {
    return cloud_layers_impl(ray, distance, steps, jitter, false);
}

fn cloud_volume(ray: vec3<f32>, distance: f32, steps: u32, jitter: f32) -> vec4<f32> {
    let result = cloud_layers_impl(ray, distance, steps, jitter, true);
    return vec4(result.color, result.transmittance);
}

// RDR2/MSFS-style sub-cloud crepuscular god rays + volumetric virga rain curtains:
// marches 8 jittered steps below the cloud deck, rendering sunlit cloud-break
// shafts and wind-slanted precipitation curtains hanging beneath dense storm cores.
fn cloud_sunbeams(ray: vec3<f32>, jitter: f32) -> vec3<f32> {
    if cloud_cover <= 0.04 || ray.y < -0.10 {
        return vec3(0.0);
    }
    let stratus = cumulus_stratus();
    let humidity = cloud_weather().y;
    let virga_amt = clamp(frame.rain * 0.90 + frame.thunder * 0.45
        + smoothstep(0.58, 0.92, stratus) * smoothstep(0.65, 0.92, humidity) * 0.35, 0.0, 1.0);
    let sun_ok = frame.light_direction.y >= 0.055;
    if !sun_ok && virga_amt <= 0.05 {
        return vec3(0.0);
    }
    let extent = min(1050.0, max(0.0, (CLOUD_BASE - frame.camera_position.y) / max(0.008, max(ray.y, 0.008))));
    if extent <= 20.0 {
        return vec3(0.0);
    }
    let cos_vl = clamp(dot(ray, frame.light_direction), -1.0, 1.0);
    var illumination = 0.0;
    var shadow_sum = 0.0;
    var virga_sum = 0.0;
    const BEAM_STEPS: u32 = 8u;
    let inv_steps = 1.0 / f32(BEAM_STEPS);
    let wind_slant = normalize(cloud_drift(1.0, 0u) + vec2(0.001, 0.0)) * (0.28 + 0.35 * weather_wind());
    for (var i = 0u; i < BEAM_STEPS; i += 1u) {
        let t0 = f32(i) * inv_steps;
        let t1 = f32(i + 1u) * inv_steps;
        let tm = mix(t0, t1, jitter);
        let d = extent * mix(tm, tm * tm, 0.55);
        let w0 = mix(t0, t0 * t0, 0.55);
        let w1 = mix(t1, t1 * t1, 0.55);
        let dt = w1 - w0;
        let pos = ray * d;
        let visibility = projected_cloud_shadow(pos);
        // Remap cloud shadow floor (~0.22..1.0) into crisp sunlit shafts vs dark shadowed columns.
        let shaft_vis = smoothstep(0.32, 0.90, visibility);
        let shaft = shaft_vis * shaft_vis;
        let world_y = frame.camera_position.y + pos.y;
        let alt_fade = exp(-max(world_y - 63.0, 0.0) * 0.0045) * smoothstep(0.0, 20.0, CLOUD_BASE - world_y);
        let dist_fade = (1.0 - exp(-d * 0.012)) * exp(-d * 0.0011);
        illumination += shaft * alt_fade * dist_fade * dt;
        shadow_sum += (1.0 - visibility) * dt;

        // Volumetric precipitation shafts (virga) hanging beneath thick storm cores.
        if virga_amt > 0.05 && d > 95.0 {
            let drop = max(0.0, CLOUD_BASE - world_y);
            let rain_core = smoothstep(0.55, 0.90, 1.0 - visibility);
            let streak_xz = frame.camera_position.xz + pos.xz - wind_slant * drop;
            let streaks = smoothstep(0.32, 0.78, cloud_noise(streak_xz * 0.0075 + vec2(0.0, frame.time * 0.04)).x);
            let vert_fade = exp(-drop * 0.011) * smoothstep(2.0, 28.0, drop);
            virga_sum += rain_core * streaks * vert_fade * dist_fade * dt;
        }
    }

    var total = vec3(0.0);
    let cloud_mask = smoothstep(0.08, 0.45, cloud_cover);
    if sun_ok {
        // Only glow where actual cloud shadows frame the sunlit shafts (zero clear-sky washout).
        let cloud_contrast = smoothstep(0.03, 0.28, shadow_sum);
        if cloud_contrast > 0.001 {
            let phase = 0.55 * photon_phase(cos_vl, 0.72) + 0.35 * photon_phase(cos_vl, 0.22) + 0.012;
            let mist = mix(0.60, 1.45, clamp(humidity * 0.65 + 0.50 * stratus + 0.35 * frame.rain, 0.0, 1.0))
                * (1.0 - 0.50 * frame.thunder);
            let beam_color = mix(direct_atmosphere_light(), vec3(0.88, 0.93, 1.0) * dot(direct_atmosphere_light(), frame.luminance), 0.25 * frame.rain);
            total += beam_color * (illumination * 0.24 * phase * mist * cloud_mask * cloud_contrast);
        }
    }
    if virga_sum > 0.001 {
        let rain_phase = 0.35 + 0.65 * photon_phase(cos_vl, 0.45);
        let rain_lit = photon_ambient_irradiance() * (0.042 * vec3(0.76, 0.84, 0.95))
            + direct_atmosphere_light() * (0.055 * rain_phase * select(0.0, 1.0, sun_ok));
        total += rain_lit * (virga_sum * virga_amt * cloud_mask * 1.45);
    }
    return total;
}
