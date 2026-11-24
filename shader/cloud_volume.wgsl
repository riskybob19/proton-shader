// Integrate scattering and transmittance, with an opacity-weighted distance for
// layer ordering and wind-aware cloud history.
struct CloudResult {
    color: vec3<f32>,
    transmittance: f32,
    distance: f32,
    layer: u32,
}

fn clear_cloud() -> CloudResult {
    return CloudResult(vec3(0.0), 1.0, 0.0, 0u);
}

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

// Jendersie & d'Eon (NVIDIA, SIGGRAPH 2023) analytic large-droplet Mie fits (5 <= d <= 50 um):
// returns vec4(g_hg, g_d, alpha, w_d).
fn get_mie_parameters(d: f32) -> vec4<f32> {
    let dc = clamp(d, 5.0, 50.0);
    let g_hg = exp(-0.0990567 / (dc - 1.67154));
    let g_d = exp(-2.20679 / (dc + 3.91029) - 0.428934);
    let alpha = exp(3.62489 - 8.29288 / (dc + 5.52825));
    let w_d = exp(-0.599085 / (dc - 0.641583) - 0.665888);
    return vec4(g_hg, g_d, alpha, w_d);
}

fn cloud_phase_mie(cosine: f32, droplet_d: f32) -> f32 {
    let p = get_mie_parameters(droplet_d);
    // Clamp ultra-narrow forward diffraction peak for finite-step real-time integration
    // (prevents single-texel solar bloom fireflies) while preserving exact Draine lobe + weight.
    let forward_hg = photon_phase(cosine, min(p.x, 0.925));
    let draine = draine_phase(cosine, p.y, p.z);
    let mie = mix(forward_hg, draine, clamp(p.w, 0.0, 1.0));
    let glory_peak = 0.065 * exp(-18.0 * (1.0 + cosine));
    let bow_d = cosine + 0.788;
    let cloudbow = 0.040 * exp(-34.0 * bow_d * bow_d);
    return mie * 0.96 + glory_peak + cloudbow;
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

// Zero-spill mat4x4/mat4x2 Neural Cloud Multi-Scattering MLP v3 (8 -> 16 -> 16 -> 2, ReLU).
// Trained on 72,000 heterogeneous 3D Nubis3 cumulus states (36.8M Woodcock delta-tracking rays,
// 16 bounces, SIGGRAPH 2023 Draine+HG Mie phase). Val R2_log=(0.7772, 0.7812), R2_lin=(0.7428, 0.7621).
fn neural_cloud_multi_scatter(tau_sun: f32, tau_sky: f32, tau_gnd: f32,
    cosine: f32, density: f32, height: f32, dim_profile: f32) -> vec2<f32> {
    let x0 = vec4(exp(-0.45 * tau_sun), 1.0 / (1.0 + 0.22 * tau_sun), exp(-0.50 * tau_sky), dim_profile);
    let x1 = vec4(exp(-0.40 * tau_gnd), cosine, 1.0 - exp(-2.5 * density), height);
    let h1_0 = max(mat4x4<f32>(
        vec4(-0.54875, -0.49192, -0.01899, -0.59268),
        vec4(0.25384, -0.65847, -1.70137, 0.52274),
        vec4(1.23655, 0.27485, -0.42259, 0.11920),
        vec4(0.37326, 0.29476, 0.01307, -0.08753)
    ) * x0 + mat4x4<f32>(
        vec4(0.42072, 0.16180, -0.08231, 0.02654),
        vec4(0.38027, 0.13250, 0.10738, 0.86923),
        vec4(-0.38427, 0.84954, 0.27589, 0.18616),
        vec4(-0.04296, -0.31757, 0.27767, -0.20990)
    ) * x1 + vec4(-0.33366, 0.40981, 0.44417, -0.56237), vec4(0.0));
    let h1_1 = max(mat4x4<f32>(
        vec4(0.76318, -0.28049, 0.33595, -0.91086),
        vec4(0.72280, 0.38875, 0.43715, 0.33573),
        vec4(-0.05504, 0.25087, -0.48003, -0.01071),
        vec4(0.21615, 0.56659, 0.08788, -0.37635)
    ) * x0 + mat4x4<f32>(
        vec4(-0.47222, 0.12661, 0.35958, 0.36604),
        vec4(-0.15377, -0.33387, 0.08905, -0.48264),
        vec4(0.65388, 0.55097, -0.02809, 0.05327),
        vec4(-0.75242, 0.00318, 1.16954, 0.14911)
    ) * x1 + vec4(-0.26255, -0.07314, -0.25252, 0.34872), vec4(0.0));
    let h1_2 = max(mat4x4<f32>(
        vec4(-1.34521, 0.59386, -0.32089, 0.74469),
        vec4(0.39881, 0.28837, 0.94789, 0.34300),
        vec4(-0.11634, 0.23306, -0.31553, 0.27293),
        vec4(-0.12550, -0.13289, 0.11880, -0.42599)
    ) * x0 + mat4x4<f32>(
        vec4(0.41780, 0.57111, -0.23474, 0.61225),
        vec4(0.69771, 0.08724, 0.21989, 0.27951),
        vec4(0.43950, 0.00872, 0.76175, 0.02056),
        vec4(0.91844, 0.53469, -0.13865, -0.14027)
    ) * x1 + vec4(0.21561, -0.20278, -0.06911, -0.32816), vec4(0.0));
    let h1_3 = max(mat4x4<f32>(
        vec4(0.27103, 0.40479, 0.07536, 1.37913),
        vec4(0.04330, 0.80615, -0.32632, -0.14081),
        vec4(0.04862, -0.52277, 0.54984, -4.31595),
        vec4(0.22630, 0.09415, 0.05470, 0.17278)
    ) * x0 + mat4x4<f32>(
        vec4(-4.72970, -0.85593, 0.12713, 0.02362),
        vec4(-0.08023, 0.09666, 0.05918, 0.00048),
        vec4(0.63080, 0.58796, -0.28254, 0.08117),
        vec4(-0.26110, -0.10586, 1.23792, -0.05791)
    ) * x1 + vec4(-0.07299, -0.11486, -0.09839, 0.02663), vec4(0.0));
    let h2_0 = max(mat4x4<f32>(
        vec4(0.17974, 0.39404, 0.75437, -0.44103),
        vec4(0.27945, -0.74945, 0.09381, -0.34379),
        vec4(-0.66494, -0.22998, 0.02392, -0.05758),
        vec4(-0.41400, 0.44328, -0.47464, 0.01495)
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.06374, -1.14568, 0.07860, -0.57594),
        vec4(-0.02242, 0.27946, -0.14832, 0.21179),
        vec4(0.65839, -0.16345, -0.24729, 0.11038),
        vec4(-0.04792, -0.34266, 0.51554, -0.02456)
    ) * h1_1 + mat4x4<f32>(
        vec4(-0.65537, 0.14874, -0.15478, 0.41945),
        vec4(0.22697, -0.05519, 0.03694, -0.76766),
        vec4(0.62061, 0.69643, 0.19282, -0.60885),
        vec4(-0.27946, 0.29741, 0.49665, -0.10783)
    ) * h1_2 + mat4x4<f32>(
        vec4(-0.43233, 0.51118, -0.16251, 1.79801),
        vec4(-0.05426, 0.26416, 0.62552, 1.17896),
        vec4(-0.06093, -0.32276, -0.22951, -0.45880),
        vec4(0.42860, -0.04246, 0.90701, -0.69595)
    ) * h1_3 + vec4(-0.38098, -0.13756, -0.03397, 0.01569), vec4(0.0));
    let h2_1 = max(mat4x4<f32>(
        vec4(-0.33922, 0.10462, 0.26904, 0.06765),
        vec4(-0.52782, -0.55254, -0.11202, -0.16241),
        vec4(-0.25358, 0.06259, 0.03064, -2.64401),
        vec4(-1.39453, -0.48584, 0.13335, -0.42449)
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.60904, 0.01399, -0.95964, -0.35225),
        vec4(0.11202, 0.09394, -0.27610, -1.24960),
        vec4(-0.21484, -0.59365, 0.51836, 0.01957),
        vec4(0.36465, 0.14998, 0.22341, 0.78530)
    ) * h1_1 + mat4x4<f32>(
        vec4(0.40416, 0.20260, -0.14419, -0.43507),
        vec4(0.29658, 0.44257, -0.14468, 0.26588),
        vec4(0.04437, -0.35792, 1.06430, -0.02255),
        vec4(-0.32683, -0.09601, 0.16212, 0.13937)
    ) * h1_2 + mat4x4<f32>(
        vec4(-1.79979, -0.20030, 0.10181, -0.88193),
        vec4(-1.15095, -0.83372, 0.44253, 0.02727),
        vec4(0.60926, -0.51654, -0.46975, 0.75065),
        vec4(-1.07271, 0.46135, 0.64868, -1.15101)
    ) * h1_3 + vec4(0.00150, 0.21431, -0.25178, -0.21174), vec4(0.0));
    let h2_2 = max(mat4x4<f32>(
        vec4(0.04342, 0.50921, -0.05405, -0.43471),
        vec4(-0.60579, 0.06640, -0.57198, 0.21799),
        vec4(-1.27555, 0.20524, 0.46951, 0.23711),
        vec4(0.20324, -0.42893, 2.02752, 0.58113)
    ) * h1_0 + mat4x4<f32>(
        vec4(0.60940, -0.09688, -0.42498, -0.22564),
        vec4(0.14459, -0.16113, 0.26869, 0.18062),
        vec4(0.20602, -1.04571, -0.07987, 0.82333),
        vec4(-0.02066, 0.62861, -0.17005, -0.06162)
    ) * h1_1 + mat4x4<f32>(
        vec4(0.26951, -0.24838, -0.01987, -0.06317),
        vec4(-0.07082, -0.02297, 0.40146, -0.19192),
        vec4(1.05375, 0.55377, 0.20406, -0.60289),
        vec4(-0.45850, 0.70103, 0.01511, -0.33928)
    ) * h1_2 + mat4x4<f32>(
        vec4(0.55033, 0.70177, 1.13784, -0.61443),
        vec4(0.44191, 0.59388, 0.47686, -0.48443),
        vec4(-0.29755, -0.57250, -0.02568, 0.44983),
        vec4(0.11263, -0.01729, -0.03296, -0.83170)
    ) * h1_3 + vec4(-0.08405, 0.27064, -0.77930, -0.34979), vec4(0.0));
    let h2_3 = max(mat4x4<f32>(
        vec4(0.04480, 0.19838, -0.02631, 0.14767),
        vec4(-0.40620, 0.80286, -0.06870, 0.20630),
        vec4(-0.34802, 0.26876, 0.51585, -0.09479),
        vec4(-0.11635, 0.49981, 0.20396, 0.06840)
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.27316, 0.17092, 0.78902, 0.13235),
        vec4(0.15084, -0.01985, 0.42756, 0.30814),
        vec4(0.84138, -0.00258, -0.19650, 0.33666),
        vec4(-0.19560, 0.18305, -0.46168, -0.23542)
    ) * h1_1 + mat4x4<f32>(
        vec4(0.17631, 0.30573, -0.19196, 0.14361),
        vec4(0.19575, -1.03070, -0.96747, -0.11138),
        vec4(-0.02801, -0.82218, 0.34550, -0.49684),
        vec4(-0.10253, -0.21347, -0.78618, -0.24630)
    ) * h1_2 + mat4x4<f32>(
        vec4(1.07272, 1.31468, 1.23214, 0.62916),
        vec4(0.26032, 0.63442, 0.09912, -0.23543),
        vec4(0.17584, 0.02514, 0.45159, 0.84484),
        vec4(0.08599, -0.14449, -1.83177, -1.45145)
    ) * h1_3 + vec4(-0.07596, -0.10378, -0.09696, -0.01666), vec4(0.0));
    let out_log = mat4x2<f32>(
        vec2(-0.29462, 0.10967),
        vec2(0.41714, 0.07637),
        vec2(-0.01911, -0.20494),
        vec2(0.08687, -0.19660)
    ) * h2_0 + mat4x2<f32>(
        vec2(-0.16344, -0.05295),
        vec2(-0.10234, -0.01147),
        vec2(0.23442, -0.02429),
        vec2(-0.12550, -0.41590)
    ) * h2_1 + mat4x2<f32>(
        vec2(0.27811, 0.02382),
        vec2(-0.00660, 0.16913),
        vec2(0.84043, -0.03012),
        vec2(-0.04555, 0.31900)
    ) * h2_2 + mat4x2<f32>(
        vec2(0.04791, 0.22158),
        vec2(-0.23242, -0.04916),
        vec2(-0.01499, -0.25143),
        vec2(0.07437, 0.32043)
    ) * h2_3 + vec2(0.13703, 0.02881);
    return max(exp(max(out_log, vec2(0.0))) - vec2(1.0), vec2(0.0)) * (1.0 / 6.0);
}

fn cloud_scattering(density: f32, depths: vec3<f32>, transmission: f32, cosine: f32,
    phase0: f32, bounced_light: f32, height: f32, storm_weight: f32,
    dim_profile: f32, layer: CloudLayerSettings) -> vec2<f32> {
    let extinction = max(layer.extinction, 1e-4);
    let albedo_s = mix(1.0, 0.66, frame.rain);
    let base_ao = mix(mix(0.54, 0.24, storm_weight), 1.08, smoothstep(0.0, 0.72, height));

    // Nubis3 dimensional-profile lighting: dark edges on shaded crevices and
    // inner glow trapped in dense cores (replaces muddy Beer-Powder terms).
    let edge_depth = smoothstep(0.0, 0.35, dim_profile);
    let local_occ = 1.0 - exp(-density * 2.2);
    let dark_edge = mix(1.0 - 0.32 * local_occ * (1.0 - max(cosine, 0.0)), 1.0, edge_depth);
    let inner_glow = mix(0.95, 1.18, smoothstep(0.15, 0.72, dim_profile));

    if neural_clouds {
        // Exact analytic single scattering (Jendersie & d'Eon Draine+HG Mie)
        // + Kallweit-descendant Monte Carlo neural MLP multi-scattering.
        let tau = extinction * depths;
        let vis1 = exp(-tau);
        let single_sun = vis1.x * phase0 * (1.0 - 0.5 * layer.upper_shadow)
            + vis1.z * PHOTON_ISOTROPIC * bounced_light
            + 0.5 * vis1.y * PHOTON_ISOTROPIC * layer.upper_shadow;
        let single_sky = vis1.y * PHOTON_ISOTROPIC * base_ao;
        // Diffusion-regime isotropization: directional memory of the solar beam decays as
        // exp(-(1 - g) * tau_sun), keeping bright silver linings on thin edges while washing
        // out unnatural circular sun spotlights underneath deep, optically thick cloud bellies.
        let cos_ms = cosine * exp(-0.14 * tau.x);
        let deep_atten = exp(-0.055 * max(tau.x - 8.0, 0.0));
        let multi = neural_cloud_multi_scatter(tau.x, tau.y, tau.z, cos_ms, density, height, dim_profile);
        let multi_sun = multi.x * 1.05 * deep_atten * (1.0 - 0.45 * layer.upper_shadow) * dark_edge
            + vis1.z * 0.35 * PHOTON_ISOTROPIC * bounced_light;
        let multi_sky = multi.y * base_ao * inner_glow * dark_edge;
        let scatter_s1 = extinction * albedo_s;
        let scatter_ms = extinction * albedo_s * albedo_s;
        let light = vec2(scatter_s1 * single_sun + scatter_ms * multi_sun,
            scatter_s1 * single_sky + scatter_ms * multi_sky);
        return light * ((1.0 - transmission) / extinction);
    }

    // Wrenninge (Sony Pictures Imageworks, Oz 2013) 3-octave geometric-decay
    // multi-scattering fallback with Nubis3 crevice & inner-glow modulation.
    var w_albedo = albedo_s;
    var w_ext = extinction;
    var g_scale = 1.0;
    var phase = phase0;
    var light = vec2(0.0);
    for (var order = 0u; order < 3u; order += 1u) {
        let visibility = exp(-w_ext * depths);
        let energy_comp = select(1.0, 1.24, order == 2u);
        let s = w_albedo * extinction * energy_comp;
        let order_edge = select(1.0, dark_edge, order > 0u);
        let order_glow = select(1.0, inner_glow, order > 0u);
        light.x += s * order_edge * (visibility.x * phase * (1.0 - 0.5 * layer.upper_shadow)
            + visibility.z * PHOTON_ISOTROPIC * bounced_light
            + 0.5 * visibility.y * PHOTON_ISOTROPIC * layer.upper_shadow);
        light.y += s * order_edge * order_glow * visibility.y * PHOTON_ISOTROPIC * base_ao;
        if order < 2u {
            w_albedo *= 0.55;
            w_ext *= 0.45;
            g_scale *= 0.75;
            phase = mix(PHOTON_ISOTROPIC, cloud_phase_multi(cosine, vec3(0.68, 0.25, 0.30) * g_scale), g_scale);
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

fn nubis_layer_march(aux_tex: texture_2d<f32>, ray: vec3<f32>, limit: f32, steps_top: u32, jitter: f32,
    layer: CloudLayerSettings, sun: vec3<f32>, ambient: vec3<f32>, haze: vec3<f32>,
    terrain_clipped: bool, fast_terrain: bool) -> CloudResult {
    let interval_limit = select(limit, max(limit, 12000.0), terrain_clipped);
    var interval = cloud_shell(ray, interval_limit, layer.bounds);
    if blocky_clouds {
        interval = cloud_slab(ray, interval_limit, layer.bounds.x, layer.bounds.y);
    }
    let start = max(interval.x, select(0.0, 3000.0 / max(length(ray.xz), 0.001), layer.kind == 1u && !blocky_clouds));
    let travel = min(2200.0, min(limit, interval.y) - start);
    if travel <= 0.0 {
        return clear_cloud();
    }
    let sample_count = mix(2.0 * f32(steps_top), f32(steps_top), abs(ray.y));
    let max_steps = u32(ceil(sample_count)) + 4u;
    var grid_travel = travel;
    if terrain_clipped {
        grid_travel = min(2200.0, max(0.0, interval.y - start));
    }
    let step_size = max(grid_travel / max(sample_count, 1.0), 1.0);
    let thickness = layer.bounds.y - layer.bounds.x;
    let inv_thickness = 1.0 / thickness;
    let light = frame.light_direction;
    let cosine = clamp(dot(ray, light), -1.0, 1.0);

    var droplet_d = 13.5;
    if layer.kind == 0u {
        droplet_d = mix(9.5, 20.0, clamp(frame.rain * 0.75 + layer.stratus * 0.35, 0.0, 1.0));
    } else if layer.kind == 1u {
        droplet_d = 19.0;
    }
    let phase0 = cloud_phase_mie(cosine, droplet_d);
    let bounced = 0.4 * max(0.0, light.y) / PHOTON_PI;
    let storm_weight = clamp(layer.stratus * 0.55 + frame.rain * 0.65 + frame.thunder * 0.35, 0.0, 1.0);

    let nvdf_meta = nvdf_read_meta(aux_tex);
    let has_nvdf = nvdf_meta.w > 0.5;
    let anchor = nvdf_meta.xy;

    var result = CloudResult(vec3(0.0), 1.0, 0.0, layer.kind);
    var weight_sum = 0.0;
    var energy = vec2(0.0);
    const MIN_TRANSMITTANCE: f32 = 0.06;

    var hit_surface = false;
    var refine_steps = 0u;
    var t = jitter * step_size;
    for (var i = 0u; i < max_steps && t < travel && result.transmittance > MIN_TRANSMITTANCE; i += 1u) {
        let active_step = select(step_size, step_size * 0.45, refine_steps > 0u);
        let along = start + t;
        var segment = min(active_step, travel - t);
        let point = ray * along;
        let world = frame.camera_position + point;
        let dist_fade = 1.0 - smoothstep(0.94, 1.0, (along - start) * (1.0 / 2200.0));

        var density = 0.0;
        var dim_profile = 0.5;
        var nvdf_sun_vis = -1.0;

        if layer.kind == 0u && !blocky_clouds {
            let radial_alt = cloud_radial_height(world.y, point.xz);
            if has_nvdf && nvdf_in_bounds(world, anchor) {
                let nvdf = nvdf_sample_trilinear(aux_tex, world, anchor);
                if nvdf.r <= 0.002 {
                    // Nubis3 conservative 3D SDF empty-space leapfrog.
                    t += clamp(nvdf.b * 0.70, step_size, step_size * 4.5);
                    continue;
                }
                let prof = nubis_dimensional_profile(world, radial_alt, cloud_coverage(world.xz),
                    layer.stratus, layer.height_scale);
                dim_profile = prof.x;
                if dim_profile <= 0.001 {
                    t += step_size;
                    continue;
                }
                nvdf_sun_vis = nvdf.a;
                density = nubis_uprez_density(world, radial_alt, dim_profile, prof.y,
                    layer.stratus, layer.height_scale) * dist_fade;
            } else {
                let prof = nubis_dimensional_profile(world, radial_alt, cloud_coverage(world.xz),
                    layer.stratus, layer.height_scale);
                dim_profile = prof.x;
                if dim_profile <= 0.001 {
                    t += step_size * 1.35;
                    continue;
                }
                density = nubis_uprez_density(world, radial_alt, dim_profile, prof.y,
                    layer.stratus, layer.height_scale) * dist_fade;
            }
        } else {
            if !blocky_clouds && volume_density_fast(point, layer) <= 0.0005 {
                t += step_size * select(1.0, 1.4, layer.kind == 1u);
                continue;
            }
            density = volume_density(point, layer) * dist_fade;
            dim_profile = clamp(density * 1.5, 0.0, 1.0);
        }

        if density > 0.001 {
            if !hit_surface && !fast_terrain {
                hit_surface = true;
                refine_steps = 4u;
                segment = min(step_size * 0.45, travel - t);
            }
            let transmission = exp(-density * segment * layer.extinction);
            let light_dither = fract(jitter + f32(i) * 0.381966);
            let sky_dither = fract(jitter + f32(i) * 0.7548777);
            let height = clamp((cloud_radial_height(world.y, point.xz) - layer.bounds.x) * inv_thickness, 0.0, 1.0);

            var light_depth = 0.0;
            var sky_depth = 0.0;
            if nvdf_sun_vis >= 0.0 {
                // Combine amortized 3D voxel solar optical depth (density-meters) with
                // a crisp local detail tap so cloud tops gleam while bellies stay deeply shaded.
                if fast_terrain {
                    light_depth = nvdf_sun_vis * 0.90 + dim_profile * 4.5;
                    sky_depth = (1.0 - height) * dim_profile * thickness * 0.55;
                } else {
                    let shadow_p = world + light * (12.0 + 14.0 * light_dither);
                    let local_dp = nubis_dimensional_profile(shadow_p, shadow_p.y,
                        cumulus_coverage(shadow_p.xz), layer.stratus, layer.height_scale).x;
                    light_depth = nvdf_sun_vis * 0.95 + local_dp * 7.5;
                    sky_depth = (1.0 - height) * dim_profile * thickness * 0.45
                        + layer_optical_depth(point, vec3(0.0, 1.0, 0.0), thickness, layer, 1u, sky_dither) * 0.55;
                }
            } else {
                let sun_taps = select(select(2u, 1u, fast_terrain), 2u, blocky_clouds);
                light_depth = layer_optical_depth(point, light, thickness, layer, sun_taps, light_dither);
                sky_depth = select(
                    layer_optical_depth(point, vec3(0.0, 1.0, 0.0), thickness, layer, 1u, sky_dither),
                    (1.0 - height) * dim_profile * thickness * 0.42,
                    fast_terrain,
                );
            }

            let ground_depth = mix(density, 1.0, max(0.0, height * 2.0 - 1.0)) * height * thickness;
            energy += result.transmittance * cloud_scattering(
                density, vec3(light_depth, sky_depth, ground_depth),
                transmission, cosine, phase0, bounced, height, storm_weight, dim_profile, layer,
            );
            let visible_opacity = result.transmittance * (1.0 - transmission);
            result.distance += along * visible_opacity;
            weight_sum += visible_opacity;
            result.transmittance *= transmission;
        }
        if refine_steps > 0u {
            t += step_size * 0.45;
            refine_steps -= 1u;
        } else {
            t += step_size;
        }
    }

    if weight_sum <= 0.0 {
        return clear_cloud();
    }
    result.transmittance = cloud_linear(MIN_TRANSMITTANCE, 1.0, result.transmittance);
    result.distance /= weight_sum;

    // RDR2 warm sunset/sunrise cloud under-lighting (locked to sun_direction).
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

// Upper-tropospheric jet-stream basis (flow, cross) gently bent by synoptic Rossby waves
// so cirrus filaments arc gracefully across the sky instead of running in straight stripes.
fn cirrus_jet_basis(world: vec2<f32>) -> vec4<f32> {
    let base_flow = normalize(cloud_drift(1.0, 3u) + vec2(0.01, 0.004));
    let rossby = 0.34 * sin(world.x * 0.00011 + cos(world.y * 0.00009) * 1.4)
        + 0.16 * cos(world.y * 0.00017 - world.x * 0.00008);
    let c = cos(rossby);
    let s = sin(rossby);
    let flow = vec2(base_flow.x * c - base_flow.y * s, base_flow.x * s + base_flow.y * c);
    let cross_v = vec2(-flow.y, flow.x);
    return vec4(flow, cross_v);
}

// Scientifically grounded 3-species cirrus density at normalized shell height h in [0, 1]:
//  1) Cirrus fibratus / uncinus (wind-sheared ice-crystal fallstreaks + generating heads)
//  2) Cirrocumulus undulatus (high-altitude Kelvin-Helmholtz gravity-wave mackerel ripples)
//  3) Cirrostratus nebulosus (optically thin halo-bearing high-altitude ice veil)
// Returns vec3(total_density, fibratus_head_weight, crystal_halo_weight).
fn cirrus_slice_density(world_xz: vec2<f32>, h: f32, cirrus: f32, patches: f32, veil: f32, fine_lod: f32) -> vec3<f32> {
    let drift = cloud_drift(frame.time, 3u);
    let basis = cirrus_jet_basis(world_xz + drift * 0.35);
    let flow = basis.xy;
    let cross_v = basis.zw;

    // Vertical wind-shear fallstreak curvature (Cirrus uncinus "mare's tails"):
    // Ice crystals nucleate in upper generating heads (h ~ 0.85) and trail downwind
    // along a curved trajectory as they fall through vertical wind shear.
    let fall = clamp(1.0 - h, 0.0, 1.0);
    let shear_shift = -flow * (210.0 * pow(fall, 1.35)) + cross_v * (42.0 * pow(fall, 1.85));
    let p = world_xz + drift + shear_shift;

    // Transform into jet-stream aligned coordinates (u = along wind, v = across wind).
    let u = dot(p, flow);
    let v = dot(p, cross_v);

    // Broad synoptic moisture corridors (smooth, organic envelopes with zero stepped borders).
    let warp_n = photon_cloud_noise(vec2(u * 0.000028, v * 0.000055)).rg * 2.0 - 1.0;
    let env_u = u * 0.000018 + warp_n.x * 0.14;
    let env_v = v * 0.000052 + warp_n.y * 0.14;
    let macro_band = 0.62 * photon_cloud_noise(vec2(env_u, env_v)).r
        + 0.38 * photon_cloud_noise(vec2(env_u * 2.3 + 0.17, env_v * 2.1 - 0.11)).r;
    let synoptic = smoothstep(clamp(0.64 - 0.52 * cirrus, 0.16, 0.68), 0.90, macro_band);

    // Quick clear-sky early-out when neither cirrus bands, cirrocumulus, nor veil are present.
    // Cirrostratus nebulosus forms in synoptic humid sheets rather than blanketing clear blue sky.
    let veil_raw = veil * smoothstep(0.48, 0.90, macro_band * 0.82 + 0.18 * warp_n.x + 0.08);
    if synoptic <= 0.002 && patches <= 0.02 && veil_raw <= 0.004 {
        return vec3(0.0);
    }

    // Bounded gentle curl bending (keeps silky strands coherent without chaotic smear).
    let curl = cloud_curl(vec2(u * 0.00016, v * 0.00034));
    let fu = u + curl.x * 85.0;
    let fv = v + curl.y * 55.0;

    // 1) Cirrus fibratus & uncinus: anisotropic high-aspect-ratio (6:1..8:1) silky ice fibers.
    var fibratus = 0.0;
    var head_w = 0.0;
    if synoptic > 0.002 {
        // Primary fibrous plumes (stretched along jet stream) + crisp silky hair filaments.
        let s1 = photon_cloud_noise(vec2(fu * 0.000090, fv * 0.00056)).r;
        let s2 = photon_cloud_noise(vec2(fu * 0.00021 + 0.31, fv * 0.00145 + s1 * 0.18)).r;
        var strands = 0.50 * s1 + 0.50 * s2;
        if fine_lod > 0.05 {
            let s3 = photon_cloud_noise(vec2(fu * 0.00048 - 0.19, fv * 0.0034 + s2 * 0.15)).g;
            let s4 = photon_cloud_noise(vec2(fu * 0.00120 + 0.43, fv * 0.0078)).r;
            let silky = smoothstep(0.18, 0.88, 0.58 * s3 + 0.42 * s4);
            strands = mix(strands, strands * (0.42 + 0.98 * silky), fine_lod * 0.92);
        }
        // Upper generating cells (cirrus uncinus heads): denser, tighter tufts near h = 0.84.
        let head_noise = photon_cloud_noise(vec2(fu * 0.00032, fv * 0.00056) + curl * 0.025).a;
        let head_profile = smoothstep(0.46, 0.88, h);
        let head = smoothstep(0.46, 0.84, head_noise) * head_profile;
        // Lower fallstreak feathering: strands thin gracefully into wispy tails toward base.
        let streak_profile = smoothstep(0.0, 0.30, h) * (1.0 - 0.30 * smoothstep(0.76, 1.0, h));
        let fiber_core = smoothstep(0.28, 0.84, strands) * streak_profile;
        let shaped = pow(synoptic, 1.25) * (fiber_core * 1.18 + head * 0.85);
        fibratus = shaped * (0.72 + 0.62 * cirrus);
        head_w = head * synoptic;
    }

    // 2) Cirrocumulus undulatus: delicate gravity-wave "mackerel sky" rippled wavelets.
    var mackerel = 0.0;
    if patches > 0.02 {
        let cc_env = photon_cloud_noise(vec2(u * 0.000034 + 0.27, v * 0.000046 - 0.19)).a;
        let cc_mask = smoothstep(clamp(0.74 - 0.46 * patches, 0.24, 0.82), 0.94, cc_env);
        if cc_mask > 0.002 {
            // Kelvin-Helmholtz wave crests perpendicular to upper-level wind shear.
            let wave_phase = u * 0.0125 + warp_n.x * 3.2 + curl.y * 1.6;
            let kh_wave = 0.5 + 0.5 * sin(wave_phase);
            let cell_a = photon_cloud_noise(vec2(fu * 0.00088, fv * 0.00098)).a;
            let cell_b = photon_cloud_noise(vec2(fu * 0.0025, fv * 0.0028)).g;
            let puffs = smoothstep(0.32, 0.80, cell_a * (0.58 + 0.58 * kh_wave) - 0.26 * cell_b * fine_lod);
            let cc_vert = smoothstep(0.30, 0.62, h) * smoothstep(1.0, 0.82, h);
            mackerel = cc_mask * puffs * puffs * cc_vert * (0.78 * patches);
        }
    }

    // 3) Cirrostratus nebulosus: synoptic translucent high-altitude ice sheet (ideal 22-deg halo host).
    let veil_vert = smoothstep(0.05, 0.40, h) * smoothstep(1.0, 0.68, h);
    let cirrostratus = veil_raw * veil_vert * 0.085;

    let total = fibratus + mackerel + cirrostratus;
    // Randomly oriented hexagonal columns/plates in cirrostratus & smooth fibratus produce
    // crisp 22-degree ice halos; turbulent cirrocumulus droplets/aggregates produce less.
    let halo_weight = clamp((cirrostratus * 2.8 + fibratus * 0.85) / max(total, 1e-4), 0.0, 1.0);
    return vec3(total, head_w, halo_weight);
}

// Physically based spectral RGB phase function for hexagonal ice crystals (columns & plates,
// refractive index n_R = 1.3078, n_G = 1.3110, n_B = 1.3157):
//  - Strong diffraction forward peak + smooth ice-facet side/backscatter continuum
//  - Chromatic 22-degree minimum-deviation halo (sharp red inner rim, soft cool outer wash)
//  - Subtle parhelia (sun dogs) on horizontally oriented ice plates at low-to-mid solar elevation
fn cirrus_ice_phase(cosine: f32, ray: vec3<f32>, halo_purity: f32) -> vec3<f32> {
    // Yang/Baum hexagonal ice-crystal phase continuum (strong forward diffraction + bright side scatter).
    let p_fwd = photon_phase(cosine, 0.85);
    let p_mid = photon_phase(cosine, 0.46);
    let p_back = photon_phase(cosine, -0.32);
    let continuum = vec3(0.54 * p_fwd + 0.30 * p_mid + 0.16 * p_back);

    if halo_purity <= 0.005 || frame.sun_direction.y < -0.02 {
        return continuum;
    }

    // Exact minimum deviation through 60-degree hexagonal ice prism faces (radians):
    // theta_min = 2 * asin(n * 0.5) - pi/3 -> R: 21.66 deg, G: 21.84 deg, B: 22.12 deg.
    let theta = acos(clamp(cosine, -1.0, 1.0));
    let theta_min = vec3(0.3780, 0.3812, 0.3861);
    let d_ang = theta - theta_min;
    // Physically asymmetric halo: sharp dark inner cutoff (no rays < theta_min), softer outer tail.
    let inv_w = select(vec3(1.0 / 0.030), vec3(1.0 / 0.0072), d_ang < vec3(0.0));
    let x = d_ang * inv_w;
    let halo22 = exp(-0.5 * x * x) * vec3(1.12, 1.00, 0.90);

    // Subtle Parhelia (Sun Dogs): horizontally oriented hexagonal plates refract light along the
    // parhelic circle (ray.y ~ sun.y) slightly outside 22 degrees as solar elevation increases.
    let sun_el = max(frame.sun_direction.y, 0.0);
    let parhelion_el = smoothstep(0.01, 0.10, sun_el) * (1.0 - smoothstep(0.42, 0.68, sun_el));
    let el_band = exp(-pow((ray.y - frame.sun_direction.y) * (1.0 / 0.026), 2.0));
    let dog_shift = 0.14 * sun_el * sun_el;
    let d_dog = theta - (theta_min + vec3(dog_shift));
    let dog_inv_w = select(vec3(1.0 / 0.026), vec3(1.0 / 0.0075), d_dog < vec3(0.0));
    let xd = d_dog * dog_inv_w;
    let sundog = exp(-0.5 * xd * xd) * el_band * parhelion_el * vec3(1.25, 1.02, 0.84);

    let sun_ok = smoothstep(-0.02, 0.08, frame.sun_direction.y) * (1.0 - 0.85 * frame.rain);
    return continuum + (halo22 * 0.24 + sundog * 0.34) * (halo_purity * sun_ok);
}

fn cirrus_weather_params() -> vec3<f32> {
    let humidity = cloud_weather().y;
    let temperature = cloud_weather().x;
    let cover_scale = cloud_cover / 0.55;
    let cirrus = cover_scale * (0.52 * (0.62 - 0.38 * cloud_square(cloud_linear(0.5, 0.9, temperature))
        + 0.38 * (1.0 - cloud_linear(0.0, 0.2, temperature))) * (1.0 - 0.42 * cloud_linear(0.5, 0.78, humidity))
        + 0.45 * frame.rain);
    let patches = 0.82 * cover_scale * photon_dampen((1.0 - 0.3 * cloud_linear(0.5, 1.0, temperature))
        * (0.5 + 0.5 * cloud_linear(0.4, 1.0, humidity)) * pow(weather_wind(), 1.35));
    let veil = cover_scale * clamp(0.06 + 0.68 * smoothstep(0.44, 0.92, humidity) + 0.50 * frame.rain, 0.0, 1.0);
    return vec3(cirrus, patches, veil);
}

fn cloud_sheet(ray: vec3<f32>, distance: f32, ambient: vec3<f32>, sun: vec3<f32>, haze: vec3<f32>) -> CloudResult {
    let interval = cloud_shell(ray, max(distance, 12000.0), vec2(588.0, 738.0));
    let visible = min(distance, interval.y) - interval.x;
    if visible <= 0.0 {
        return clear_cloud();
    }
    let span = max(interval.y - interval.x, 1.0);
    let vis_ratio = clamp(visible / span, 0.0, 1.0);
    let path_len = min(420.0, span) * vis_ratio;
    let wp = cirrus_weather_params();

    // 3-stratum volumetric ice-crystal march (bottom virga tails, mid fallstreaks, top generating heads)
    // ordered front-to-back relative to camera altitude for true 3D fallstreak parallax & self-shadowing.
    let from_above = frame.camera_position.y > 663.0;
    let heights = array<f32, 3>(0.18, 0.50, 0.84);
    let seg_len = path_len * (1.0 / 3.0);
    let light = frame.light_direction;
    let cosine = clamp(dot(ray, light), -1.0, 1.0);
    let sun_horiz = light.xz / max(abs(light.y), 0.16);

    // High-altitude Earth-curvature twilight alpenglow (9-12 km ice crystals stay bathed in
    // warm amber/coral/magenta sunlight after lower clouds drop into Earth's shadow).
    let sun_y = frame.sun_direction.y;
    let golden = smoothstep(-0.08, 0.02, sun_y) * (1.0 - smoothstep(0.05, 0.22, sun_y)) * (1.0 - 0.85 * frame.rain);
    let afterglow = smoothstep(-0.085, -0.02, sun_y) * (1.0 - smoothstep(-0.01, 0.07, sun_y)) * (1.0 - 0.85 * frame.rain);
    let cirrus_sun_tint = mix(vec3(1.0), vec3(1.24, 0.78, 0.54), golden * 0.65)
        + vec3(0.42, 0.14, 0.22) * (afterglow * 0.45);
    let cirrus_amb_tint = mix(vec3(0.94, 0.98, 1.06), vec3(0.74, 0.80, 1.18), golden * 0.45 + afterglow * 0.65);

    var trans = 1.0;
    var accum = vec3(0.0);
    var dist_sum = 0.0;
    var w_sum = 0.0;
    var tau_total = 0.0;

    for (var s = 0u; s < 3u; s += 1u) {
        let idx = select(s, 2u - s, from_above);
        let h = heights[idx];
        let along = mix(interval.x, interval.y, select(h, 1.0 - h, from_above));
        if along > distance {
            continue;
        }
        let world = frame.camera_position + ray * along;
        let horiz_d = length(world.xz - frame.camera_position.xz);
        let fine_lod = 1.0 - smoothstep(2200.0, 7500.0, horiz_d);

        let sample_v = cirrus_slice_density(world.xz, h, wp.x, wp.y, wp.z, fine_lod);
        let dens = sample_v.x;
        if dens <= 0.0005 {
            continue;
        }

        let slice_tau = dens * seg_len * 0.0175;
        let slice_tr = exp(-slice_tau);
        tau_total += slice_tau;

        // Solar self-shadowing from upper generating heads onto lower sheared fallstreaks.
        var tau_sun = dens * (1.0 - h) * 1.4;
        if h < 0.80 {
            let shadow_xz = world.xz + clamp(sun_horiz * ((0.84 - h) * 150.0), vec2(-220.0), vec2(220.0));
            let upper_d = cirrus_slice_density(shadow_xz, min(h + 0.34, 0.92), wp.x, wp.y, wp.z, 0.0).x;
            tau_sun += upper_d * 2.2;
        }

        // Single-scatter hexagonal ice phase (halo peaks in optically thin ice, washes out in thick cores).
        let thin_halo_ok = exp(-tau_total * 1.35);
        let halo_purity = sample_v.z * thin_halo_ok;
        let ice_phase = cirrus_ice_phase(cosine, ray, halo_purity);

        // 1st-order directional ice refraction/diffraction + 2nd-order diffuse ice multiple scattering
        // (ice albedo ~ 0.99999 in visible wavelengths -> luminous pure ivory-white filaments).
        let direct_1 = ice_phase * exp(-1.55 * tau_sun) * 0.42;
        let direct_2 = vec3(0.068 * exp(-0.42 * tau_sun) * (0.75 + 0.35 * sample_v.y));
        let sky_occ = mix(0.68, 1.05, h) * (0.45 + 0.55 * exp(-1.1 * tau_sun));
        let scatter_rgb = sun * cirrus_sun_tint * (direct_1 + direct_2)
            + ambient * cirrus_amb_tint * (0.125 * sky_occ);

        let vis_w = trans * (1.0 - slice_tr);
        accum += scatter_rgb * vis_w;
        dist_sum += along * vis_w;
        w_sum += vis_w;
        trans *= slice_tr;
    }

    if w_sum <= 1e-5 {
        return clear_cloud();
    }
    let avg_dist = dist_sum / w_sum;
    return CloudResult(cloud_aerial(accum, trans, ray, avg_dist, haze), trans, avg_dist, 3u);
}

fn noctilucent_sheet(ray: vec3<f32>, distance: f32) -> CloudResult {
    let twilight = smoothstep(0.025, 0.11, -frame.sun_direction.y)
        * (1.0 - smoothstep(0.14, 0.32, -frame.sun_direction.y)) * (1.0 - frame.rain);
    if twilight < 0.001 || ray.y <= 0.01 || (frame.moon_phase % 3u) == 0u {
        return clear_cloud();
    }
    let along = (775.0 - frame.camera_position.y) / ray.y;
    if along <= 0.0 || along >= distance {
        return clear_cloud();
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

fn cloud_layers_impl(aux_tex: texture_2d<f32>, ray: vec3<f32>, distance: f32, steps: u32,
    jitter: f32, terrain_clipped: bool, fast_terrain: bool) -> CloudResult {
    if !volumetric_clouds || cloud_cover <= 0.0 {
        return clear_cloud();
    }
    let sun = direct_atmosphere_light();
    let ambient = photon_ambient_irradiance() + storm_flash() * vec3(0.9, 1.6, 3.0);
    let haze = atmosphere(ray);
    if blocky_clouds {
        let first = CloudLayerSettings(vec2(CLOUD_LAYER_HEIGHT - 8.0, CLOUD_LAYER_HEIGHT + 8.0),
            0u, 0.0, vec2(0.0), 1.0, 0.19, 0.0, 0.0);
        let base = nubis_layer_march(aux_tex, ray, distance, steps, jitter, first,
            sun, ambient, haze, terrain_clipped, fast_terrain);
        if second_blocky_layer {
            let second = CloudLayerSettings(vec2(BLOCKY_SECOND_HEIGHT - 8.0, BLOCKY_SECOND_HEIGHT + 8.0),
                1u, 0.0, vec2(0.0), 1.0, 0.19, 0.0, 0.0);
            return blend_clouds(base, nubis_layer_march(aux_tex, ray, distance, steps, jitter,
                second, sun, ambient, haze, terrain_clipped, fast_terrain));
        }
        return base;
    }

    let stratus = cumulus_stratus();
    let mid_blend = mid_stratus();
    let mid_cover = mid_coverage(mid_blend);
    let upper_shadow = cloud_linear(0.7, 1.0, dot(mid_cover, vec2(0.25, 0.75)) * (1.0 + mid_blend))
        * photon_dampen(smoothstep(-0.05, 0.08, frame.sun_direction.y));
    let cover = cumulus_coverage_range(stratus);
    let extinction = mix(0.68, 1.15, smoothstep(0.0, 0.3, abs(frame.sun_direction.y)))
        * (1.0 - 0.33 * frame.rain) * (1.0 - 0.6 * upper_shadow)
        * (1.0 - 0.4 * cloud_linear(0.8, 1.0, cover.y * (1.0 - stratus))) * (1.0 - 0.5 * stratus);
    let low_sun = cloud_altitude_sun(sun, 0.5 * (CLOUD_BASE + CLOUD_TOP));
    let low = CloudLayerSettings(vec2(CLOUD_BASE, CLOUD_TOP), 0u, stratus, cover,
        cumulus_height_scale(stratus), extinction, upper_shadow, 0.0);
    var result = nubis_layer_march(aux_tex, ray, distance, steps, jitter, low,
        low_sun, ambient, haze, terrain_clipped, fast_terrain);
    if result.transmittance <= 0.001 && frame.camera_position.y < CLOUD_BASE && result.distance < 3000.0 {
        return result;
    }

    let ray_end_y = frame.camera_position.y + ray.y * distance;
    if mid_cover.y > 0.001 && (!fast_terrain || ray_end_y > MID_CLOUD_BASE || frame.camera_position.y > MID_CLOUD_BASE) {
        let mid_sun = cloud_altitude_sun(sun, 0.5 * (MID_CLOUD_BASE + MID_CLOUD_TOP));
        let middle = CloudLayerSettings(vec2(MID_CLOUD_BASE, MID_CLOUD_TOP), 2u, mid_blend,
            mid_cover, 1.0, 0.60, 0.0, 0.0);
        let mid_steps = select(max(7u, steps / 2u), 5u, fast_terrain);
        result = blend_clouds(result, nubis_layer_march(aux_tex, ray, distance, mid_steps,
            jitter, middle, mid_sun, ambient, haze, terrain_clipped, fast_terrain));
    }
    if !fast_terrain {
        let amount = congestus_amount();
        if amount > 0.001 && distance > 3000.0 && result.transmittance > 0.02 {
            let tower_sun = cloud_altitude_sun(sun, 0.5 * (CONGESTUS_BASE + CONGESTUS_TOP));
            let tower = CloudLayerSettings(vec2(CONGESTUS_BASE, CONGESTUS_TOP), 1u, 0.0, vec2(0.0),
                1.0, 0.8, 0.0, amount);
            let distant = nubis_layer_march(aux_tex, ray, distance, max(10u, steps / 2u), jitter,
                tower, tower_sun, ambient, haze, terrain_clipped, false);
            let fade = mix(1.0, distant.transmittance, cloud_linear(2250.0, 3000.0, result.distance));
            result.color *= fade;
            result.transmittance = mix(1.0, result.transmittance, fade);
            result = blend_clouds(result, distant);
        }
    }
    if result.transmittance <= 0.001 && frame.camera_position.y < MID_CLOUD_BASE {
        return result;
    }
    if fast_terrain && ray_end_y < 588.0 && frame.camera_position.y < 588.0 {
        return result;
    }
    let cirrus_sun = cloud_altitude_sun(sun, 660.0);
    return blend_clouds(blend_clouds(result, cloud_sheet(ray, distance, ambient, cirrus_sun, haze)),
        noctilucent_sheet(ray, distance));
}

fn cloud_layers(aux_tex: texture_2d<f32>, ray: vec3<f32>, distance: f32, steps: u32, jitter: f32) -> CloudResult {
    return cloud_layers_impl(aux_tex, ray, distance, steps, jitter, false, false);
}

fn cloud_volume_terrain(aux_tex: texture_2d<f32>, ray: vec3<f32>, distance: f32, jitter: f32) -> vec4<f32> {
    let result = cloud_layers_impl(aux_tex, ray, distance, 9u, jitter, true, true);
    return vec4(result.color, result.transmittance);
}

fn nubis_layer_transmittance(aux_tex: texture_2d<f32>, ray: vec3<f32>, limit: f32, steps_top: u32,
    jitter: f32, layer: CloudLayerSettings) -> f32 {
    let interval_limit = max(limit, 12000.0);
    var interval = cloud_shell(ray, interval_limit, layer.bounds);
    if blocky_clouds {
        interval = cloud_slab(ray, interval_limit, layer.bounds.x, layer.bounds.y);
    }
    let start = max(interval.x, select(0.0, 3000.0 / max(length(ray.xz), 0.001), layer.kind == 1u && !blocky_clouds));
    let travel = min(2200.0, min(limit, interval.y) - start);
    if travel <= 0.0 {
        return 1.0;
    }
    let sample_count = mix(2.0 * f32(steps_top), f32(steps_top), abs(ray.y));
    let max_steps = u32(ceil(sample_count));
    let grid_travel = min(2200.0, max(0.0, interval.y - start));
    let step_size = max(grid_travel / max(sample_count, 1.0), 1.0);

    let nvdf_meta = nvdf_read_meta(aux_tex);
    let has_nvdf = nvdf_meta.w > 0.5;
    let anchor = nvdf_meta.xy;

    const MIN_TRANSMITTANCE: f32 = 0.06;
    var transmittance = 1.0;
    var hit = false;
    var t = jitter * step_size;
    for (var i = 0u; i < max_steps && t < travel && transmittance > MIN_TRANSMITTANCE; i += 1u) {
        let along = start + t;
        let segment = min(step_size, travel - t);
        let point = ray * along;
        let world = frame.camera_position + point;
        let dist_fade = 1.0 - smoothstep(0.94, 1.0, (along - start) * (1.0 / 2200.0));

        var density = 0.0;
        if layer.kind == 0u && !blocky_clouds {
            let radial_alt = cloud_radial_height(world.y, point.xz);
            if has_nvdf && nvdf_in_bounds(world, anchor) {
                let nvdf = nvdf_sample_trilinear(aux_tex, world, anchor);
                if nvdf.r <= 0.002 {
                    t += clamp(nvdf.b * 0.70, step_size, step_size * 4.5);
                    continue;
                }
                let prof = nubis_dimensional_profile(world, radial_alt, cloud_coverage(world.xz),
                    layer.stratus, layer.height_scale);
                if prof.x <= 0.001 {
                    t += step_size;
                    continue;
                }
                density = nubis_uprez_density(world, radial_alt, prof.x, prof.y,
                    layer.stratus, layer.height_scale) * dist_fade;
            } else {
                let prof = nubis_dimensional_profile(world, radial_alt, cloud_coverage(world.xz),
                    layer.stratus, layer.height_scale);
                if prof.x <= 0.001 {
                    t += step_size * 1.35;
                    continue;
                }
                density = nubis_uprez_density(world, radial_alt, prof.x, prof.y,
                    layer.stratus, layer.height_scale) * dist_fade;
            }
        } else {
            if !blocky_clouds && volume_density_fast(point, layer) <= 0.0005 {
                t += step_size * select(1.0, 1.4, layer.kind == 1u);
                continue;
            }
            density = volume_density(point, layer) * dist_fade;
        }

        if density > 0.001 {
            transmittance *= exp(-density * segment * layer.extinction);
            hit = true;
        }
        t += step_size;
    }

    if !hit || transmittance >= 0.99999 {
        return 1.0;
    }
    return cloud_linear(MIN_TRANSMITTANCE, 1.0, transmittance);
}

fn cloud_sheet_transmittance(ray: vec3<f32>, distance: f32) -> f32 {
    let interval = cloud_shell(ray, max(distance, 12000.0), vec2(588.0, 738.0));
    let visible = min(distance, interval.y) - interval.x;
    if visible <= 0.0 {
        return 1.0;
    }
    let span = max(interval.y - interval.x, 1.0);
    let path_len = min(420.0, span) * clamp(visible / span, 0.0, 1.0);
    let wp = cirrus_weather_params();
    let along = 0.5 * (interval.x + interval.y);
    let world = frame.camera_position + ray * along;
    let dens = cirrus_slice_density(world.xz, 0.50, wp.x, wp.y, wp.z, 0.0).x;
    if dens <= 0.0005 {
        return 1.0;
    }
    return exp(-dens * path_len * 0.0175);
}

fn cloud_volume_transmittance_only(aux_tex: texture_2d<f32>, ray: vec3<f32>, distance: f32, jitter: f32) -> f32 {
    if !volumetric_clouds || cloud_cover <= 0.0 {
        return 1.0;
    }
    if blocky_clouds {
        let first = CloudLayerSettings(vec2(CLOUD_LAYER_HEIGHT - 8.0, CLOUD_LAYER_HEIGHT + 8.0),
            0u, 0.0, vec2(0.0), 1.0, 0.19, 0.0, 0.0);
        var tr = nubis_layer_transmittance(aux_tex, ray, distance, 9u, jitter, first);
        if second_blocky_layer {
            let second = CloudLayerSettings(vec2(BLOCKY_SECOND_HEIGHT - 8.0, BLOCKY_SECOND_HEIGHT + 8.0),
                1u, 0.0, vec2(0.0), 1.0, 0.19, 0.0, 0.0);
            tr *= nubis_layer_transmittance(aux_tex, ray, distance, 9u, jitter, second);
        }
        return tr;
    }
    let stratus = cumulus_stratus();
    let mid_blend = mid_stratus();
    let mid_cover = mid_coverage(mid_blend);
    let upper_shadow = cloud_linear(0.7, 1.0, dot(mid_cover, vec2(0.25, 0.75)) * (1.0 + mid_blend))
        * photon_dampen(smoothstep(-0.05, 0.08, frame.sun_direction.y));
    let cover = cumulus_coverage_range(stratus);
    let extinction = mix(0.68, 1.15, smoothstep(0.0, 0.3, abs(frame.sun_direction.y)))
        * (1.0 - 0.33 * frame.rain) * (1.0 - 0.6 * upper_shadow)
        * (1.0 - 0.4 * cloud_linear(0.8, 1.0, cover.y * (1.0 - stratus))) * (1.0 - 0.5 * stratus);
    let low = CloudLayerSettings(vec2(CLOUD_BASE, CLOUD_TOP), 0u, stratus, cover,
        cumulus_height_scale(stratus), extinction, upper_shadow, 0.0);
    var tr = nubis_layer_transmittance(aux_tex, ray, distance, 9u, jitter, low);
    if tr <= 0.001 && frame.camera_position.y < CLOUD_BASE {
        return tr;
    }
    let ray_end_y = frame.camera_position.y + ray.y * distance;
    if mid_cover.y > 0.001 && (ray_end_y > MID_CLOUD_BASE || frame.camera_position.y > MID_CLOUD_BASE) {
        let middle = CloudLayerSettings(vec2(MID_CLOUD_BASE, MID_CLOUD_TOP), 2u, mid_blend,
            mid_cover, 1.0, 0.60, 0.0, 0.0);
        tr *= nubis_layer_transmittance(aux_tex, ray, distance, 5u, jitter, middle);
    }
    if tr <= 0.001 && frame.camera_position.y < MID_CLOUD_BASE {
        return tr;
    }
    if ray_end_y >= 588.0 || frame.camera_position.y >= 588.0 {
        tr *= cloud_sheet_transmittance(ray, distance);
    }
    return tr;
}

// RDR2/MSFS-style sub-cloud crepuscular god rays + volumetric virga rain curtains:
// marches 8 jittered steps below the cloud deck using cached cloud_aux lookups.
fn cloud_sunbeams(aux_tex: texture_2d<f32>, ray: vec3<f32>, jitter: f32) -> vec3<f32> {
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
        var visibility = cloud_shadow_from_aux(aux_tex, pos);
        if visibility < 0.0 {
            visibility = projected_cloud_shadow(pos);
        }
        let shaft_vis = smoothstep(0.32, 0.90, visibility);
        let shaft = shaft_vis * shaft_vis;
        let world_y = frame.camera_position.y + pos.y;
        let alt_fade = exp(-max(world_y - 63.0, 0.0) * 0.0045) * smoothstep(0.0, 20.0, CLOUD_BASE - world_y);
        let dist_fade = (1.0 - exp(-d * 0.012)) * exp(-d * 0.0011);
        illumination += shaft * alt_fade * dist_fade * dt;
        shadow_sum += (1.0 - visibility) * dt;

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
