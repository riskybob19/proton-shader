override neural_water: bool = true;

struct JerlovOptics {
    absorption: vec3<f32>,
    scattering: vec3<f32>,
    extinction: vec3<f32>,
    turbidity: f32,
    cdom: f32,
}

// Derive continuous Jerlov Oceanic I..III -> Coastal 1C..7C underwater optics
// from Minecraft's live biome-blended underwater fog_color + temperature + downfall.
fn jerlov_underwater_optics() -> JerlovOptics {
    var turbidity = 0.22;
    var cdom = 0.14;
    if biome_water_tint {
        let fog_c = clamp(frame.fog_color, vec3(0.01), vec3(1.0));
        let fog_sum = max(fog_c.r + fog_c.g + fog_c.b, 0.02);
        let g_norm = fog_c.g / fog_sum;
        let b_norm = fog_c.b / fog_sum;
        // Swamps / mangroves have green/olive underwater fog (g >= b) + high humidity.
        let swamp_cue = smoothstep(0.0, 0.22, g_norm - b_norm * 0.85)
            * smoothstep(0.35, 0.85, frame.downfall + 0.25);
        // Warm tropical oceans have bright cyan-turquoise fog (balanced high G and B) & warm temp.
        let warm_cue = smoothstep(0.45, 0.90, frame.temperature)
            * smoothstep(0.28, 0.44, g_norm) * (1.0 - swamp_cue);
        // Cold / frozen oceans are deeper slate-blue.
        let cold_cue = 1.0 - smoothstep(0.05, 0.45, frame.temperature);

        turbidity = clamp(0.24 - 0.13 * warm_cue + 0.08 * cold_cue + 0.46 * swamp_cue, 0.06, 0.88);
        cdom = clamp(0.14 - 0.08 * warm_cue + 0.06 * cold_cue + 0.64 * swamp_cue, 0.02, 0.92);
    }
    let a = vec3(
        0.24 + 0.10 * turbidity + 0.08 * cdom,
        0.085 + 0.055 * turbidity + 0.095 * cdom,
        0.042 + 0.040 * turbidity + 0.240 * cdom,
    );
    let b_base = 0.016 + 0.095 * turbidity;
    let b = b_base * vec3(
        0.78 + 0.16 * cdom,
        1.02 + 0.08 * cdom,
        1.16 - 0.30 * cdom,
    );
    return JerlovOptics(a, b, a + b, turbidity, cdom);
}

// Zero-spill mat4x4/mat4x3 Multi-Turbidity Jerlov Neural Underwater Radiative Transfer MLP v2 (8 -> 16 -> 16 -> 3, ReLU).
// Trained on 36,000 Monte Carlo Jerlov I..7C ocean/coastal states (320 paths x 10 bounces = 11.5M rays, Petzold phase).
// Validation R2=0.9662, MAE=0.00070, energy ratio vs path tracer=1.0028.
fn neural_underwater_scatter(travel: f32, eye_depth: f32, ray_y: f32,
    cos_vl: f32, sun_y: f32, turbidity: f32, cdom: f32) -> vec3<f32> {
    let x0 = vec4(1.0 - exp(-0.08 * travel), exp(-0.10 * eye_depth), 1.0 / (1.0 + 0.12 * eye_depth), ray_y);
    let x1 = vec4(cos_vl, sun_y, turbidity, cdom);
    let h1_0 = max(mat4x4<f32>(
        vec4(0.04016, 0.20693, 0.19914, 0.02194),
        vec4(-0.10368, -0.19305, -0.89964, -0.35145),
        vec4(0.68932, 0.26415, -0.06522, 0.01843),
        vec4(0.13104, 0.04325, -0.15164, -0.00518)
    ) * x0 + mat4x4<f32>(
        vec4(-0.76376, 0.15705, -0.07136, 0.96038),
        vec4(0.00505, 0.04495, -0.13203, 0.10308),
        vec4(-0.23477, 0.79405, 0.66668, 0.40055),
        vec4(-0.34929, -0.74540, 0.31294, -0.33327)
    ) * x1 + vec4(-0.00533, 0.48078, 0.24742, -0.12059), vec4(0.0));
    let h1_1 = max(mat4x4<f32>(
        vec4(-0.03409, 0.07766, 0.08933, 0.32754),
        vec4(-1.21347, 0.19938, -0.26308, 0.82231),
        vec4(-0.08664, -0.29289, -0.09085, -0.58633),
        vec4(-0.14048, -0.05855, -0.69634, 1.07690)
    ) * x0 + mat4x4<f32>(
        vec4(-0.00286, -0.01295, 0.00575, -0.19457),
        vec4(-0.01554, 0.00823, -0.01624, 0.02558),
        vec4(0.02385, 0.60019, 0.06974, -0.20633),
        vec4(-0.00368, 0.61720, -0.03792, 0.01164)
    ) * x1 + vec4(1.31687, -0.14262, 0.43619, -0.38320), vec4(0.0));
    let h1_2 = max(mat4x4<f32>(
        vec4(0.01258, 0.00220, 0.18882, 0.11127),
        vec4(0.00449, 0.46089, -0.94450, 0.55271),
        vec4(-0.70301, 0.42975, -0.48123, 0.77205),
        vec4(-0.06818, 0.40437, -0.22515, 0.10625)
    ) * x0 + mat4x4<f32>(
        vec4(1.69101, 0.02885, -0.15623, -0.24330),
        vec4(0.00175, 0.00149, -0.12788, 0.01573),
        vec4(0.11158, 0.05318, -0.03614, -0.12163),
        vec4(-0.00210, -0.06502, 0.42396, -0.11050)
    ) * x1 + vec4(-0.85074, -0.83279, 0.62867, -0.78240), vec4(0.0));
    let h1_3 = max(mat4x4<f32>(
        vec4(-1.65868, -1.10503, -0.02213, 0.29112),
        vec4(-0.87711, -0.97231, -1.45601, 0.37053),
        vec4(-0.05094, -0.49727, -0.54808, 0.54779),
        vec4(-0.21120, -0.17560, -0.22313, 0.47104)
    ) * x0 + mat4x4<f32>(
        vec4(0.00289, -0.06136, 0.45531, 0.11031),
        vec4(-0.00342, -0.04323, -0.19391, -0.02430),
        vec4(0.10966, 0.00224, 0.68416, -0.09899),
        vec4(-0.04209, 0.17717, 0.13681, 0.03655)
    ) * x1 + vec4(1.17287, 1.06772, 0.27835, -0.69271), vec4(0.0));
    let h2_0 = max(mat4x4<f32>(
        vec4(-0.26915, -0.30402, -0.32162, 0.45506),
        vec4(-0.45477, -0.23774, -0.36027, -0.24195),
        vec4(-0.14284, -0.20359, 0.79962, -0.34812),
        vec4(-0.44065, 0.03244, -0.22493, 0.08451)
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.27486, -0.06530, -0.44904, -1.68990),
        vec4(-0.17569, -0.03291, -0.27530, -0.18070),
        vec4(-0.64577, -0.22106, 0.23991, -0.31156),
        vec4(-0.35608, -0.47356, 0.12195, -0.14362)
    ) * h1_1 + mat4x4<f32>(
        vec4(-0.13652, -0.76765, -0.07248, 4.10572),
        vec4(-0.00490, -0.23958, -0.04929, 0.37652),
        vec4(0.58423, -0.17405, -0.18322, -0.50158),
        vec4(0.69412, 0.43277, 0.00296, 0.23303)
    ) * h1_2 + mat4x4<f32>(
        vec4(0.07287, -0.16335, -0.23908, -0.45382),
        vec4(-0.29588, 0.16097, -0.49115, 0.42443),
        vec4(-0.75861, -0.41706, -0.38351, 0.09692),
        vec4(-0.10805, -0.55159, -0.06127, 0.01959)
    ) * h1_3 + vec4(-0.18460, -0.18464, -0.10230, -0.48874), vec4(0.0));
    let h2_1 = max(mat4x4<f32>(
        vec4(0.24476, -0.21812, -0.20495, 0.08134),
        vec4(0.90386, -0.23202, -0.25292, 0.14744),
        vec4(-0.24224, 0.25479, 0.25818, 0.00167),
        vec4(0.06053, -0.54157, 0.56199, 0.23330)
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.05938, -0.05759, -0.86158, -0.03112),
        vec4(-0.54028, -0.22845, -0.37965, 0.19388),
        vec4(0.20929, -0.11882, -0.52298, 0.05755),
        vec4(-0.16324, 0.24145, 0.04457, -0.03292)
    ) * h1_1 + mat4x4<f32>(
        vec4(0.19625, 0.22965, 1.96983, 0.56146),
        vec4(-1.16822, 0.37676, -0.06667, -1.79600),
        vec4(-0.47660, -0.13297, -0.33713, -0.71324),
        vec4(-0.61122, -0.34536, -0.45524, -0.39721)
    ) * h1_2 + mat4x4<f32>(
        vec4(-0.72321, -0.19724, -0.27025, -0.60008),
        vec4(0.67289, 0.08566, -0.17829, -0.45652),
        vec4(-0.37339, 0.04836, -0.46504, -0.77569),
        vec4(0.28146, -0.40641, -0.06056, 0.28069)
    ) * h1_3 + vec4(-0.25982, -0.12118, -0.26903, 0.07751), vec4(0.0));
    let h2_2 = max(mat4x4<f32>(
        vec4(0.09204, 0.17869, -0.11952, -0.13479),
        vec4(0.13275, 0.88205, -0.30822, -0.46645),
        vec4(-0.40102, -0.09263, 0.37757, -0.20712),
        vec4(0.21643, 0.01680, -0.03922, 0.36884)
    ) * h1_0 + mat4x4<f32>(
        vec4(-2.98997, 0.09930, 0.25577, -0.74663),
        vec4(0.16301, -0.94706, -0.16452, 0.42146),
        vec4(-0.86335, 0.07638, -0.31430, 0.24155),
        vec4(-0.69433, -0.16502, 0.02395, 0.08261)
    ) * h1_1 + mat4x4<f32>(
        vec4(-0.52882, 0.11487, -0.26433, 0.43240),
        vec4(0.51705, -1.09899, -0.05374, 0.08080),
        vec4(-0.01282, -0.17523, 0.05012, 0.21969),
        vec4(0.20360, -0.08899, 0.65923, -0.23896)
    ) * h1_2 + mat4x4<f32>(
        vec4(-2.14255, -0.38663, 1.35887, -0.08083),
        vec4(0.00003, 0.17440, -1.13778, -0.34935),
        vec4(-0.24093, -0.31402, -0.11638, 0.24848),
        vec4(0.42331, 0.05420, 0.40113, -0.50335)
    ) * h1_3 + vec4(-0.26913, -0.67267, 0.11553, -0.19689), vec4(0.0));
    let h2_3 = max(mat4x4<f32>(
        vec4(0.14269, 0.28787, 0.14630, 0.13287),
        vec4(-0.17970, 0.15929, -0.11336, 0.36126),
        vec4(0.29123, -0.31871, 0.26351, -0.32743),
        vec4(0.17904, 0.24756, 0.02820, 0.35484)
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.54676, -1.91915, 0.03506, -1.39557),
        vec4(0.13314, 0.00811, -0.11399, 0.10089),
        vec4(-0.57620, 0.26224, 0.30229, 0.17372),
        vec4(0.00486, -0.36927, 0.01066, -0.13320)
    ) * h1_1 + mat4x4<f32>(
        vec4(-0.49575, 1.63836, 0.27879, 1.06055),
        vec4(-0.12440, -0.38509, -0.99703, -0.97777),
        vec4(0.43573, 1.42651, -0.58600, 0.32679),
        vec4(0.13721, -0.96451, -0.68269, -1.04806)
    ) * h1_2 + mat4x4<f32>(
        vec4(-0.34543, -1.88076, -1.26625, -1.78048),
        vec4(-0.26213, -0.19408, 0.99275, 2.12897),
        vec4(-0.07074, -0.97815, 0.14718, -0.50416),
        vec4(-0.49567, 0.47139, -0.35651, 0.30249)
    ) * h1_3 + vec4(-0.11431, 0.15629, 0.47385, 0.33536), vec4(0.0));
    let out_log = mat4x3<f32>(
        vec3(-0.21216, -0.11377, 0.21807),
        vec3(-0.08002, 0.04711, 0.31730),
        vec3(-0.01742, 0.20695, -0.17347),
        vec3(2.28457, 2.59797, 2.42777)
    ) * h2_0 + mat4x3<f32>(
        vec3(-0.04637, 0.00687, 0.31448),
        vec3(-0.43456, -0.03787, -0.34113),
        vec3(-0.19569, 1.17715, 1.37898),
        vec3(0.01965, 0.30435, 0.07796)
    ) * h2_1 + mat4x3<f32>(
        vec3(-0.37722, -0.72151, -0.73810),
        vec3(-0.00400, 0.08251, 1.17675),
        vec3(-0.12093, -0.19655, -0.19671),
        vec3(0.10540, -0.73731, 0.51564)
    ) * h2_2 + mat4x3<f32>(
        vec3(0.32882, -0.15160, -0.51653),
        vec3(0.40634, 0.40306, 0.42385),
        vec3(-0.13022, -0.20695, -0.21454),
        vec3(0.14933, 0.30921, 0.23197)
    ) * h2_3 + vec3(0.07595, 0.12612, 0.12395);
    return max(exp(max(out_log, vec3(0.0))) - vec3(1.0), vec3(0.0)) * (1.0 / 24.0);
}

// Fast 2-iteration rotated Lagrangian caustic fold kernel aligned with water.wgsl
// so volumetric underwater god-ray curtains match the seabed caustic web.
fn underwater_caustic_curtain(coord: vec2<f32>, t: f32) -> f32 {
    let rot = mat2x2<f32>(vec2(0.9563, 0.2924), vec2(-0.2924, 0.9563));
    let p = fract((rot * coord) * 0.145) * TAU;
    var i = p;
    var sum = 1.0;
    for (var n = 0u; n < 2u; n += 1u) {
        let tn = t * (1.0 - 2.4 / f32(n + 1u));
        i = p + vec2(cos(tn - i.x) + sin(tn + i.y), sin(tn - i.y) + cos(tn + i.x));
        let sx = sin(i.x + tn);
        let cy = cos(i.y + tn);
        sum += abs(sx * cy) / (1.18 * sqrt(sx * sx + cy * cy + 1e-5));
    }
    let c = clamp(1.14 - pow(sum * 0.333333, 1.35), 0.0, 1.0);
    return c * c * c;
}

// Photon uses separate absorption and scattering below the waterline.
fn raymarch_underwater(ray: vec3<f32>, distance: f32, jitter: f32) -> AirFog {
    let travel = min(distance, 64.0);
    if travel <= 0.0 {
        return AirFog(vec3(0.0), vec3(1.0));
    }
    let jerlov = jerlov_underwater_optics();
    let extinction = jerlov.extinction;
    let scattering_rgb = jerlov.scattering;
    let access = smoothstep(0.02, 0.9, frame.eye_light_smooth.y);
    // Localized underwater blocklight scatter (sea lanterns, magma, conduits, pickles).
    let block_e = frame.eye_light_smooth.x * frame.eye_light_smooth.x;
    let local_light = vec3(0.18, 0.14, 0.095) * block_e;
    let water_fill = local_light + vec3(0.020, 0.043, 0.057) * frame.night_vision;

    // Exact Snell-refracted underwater sun direction (n_water = 1.333).
    let sun_y = clamp(frame.light_direction.y, 0.05, 1.0);
    let refr_sun = normalize(vec3(
        frame.light_direction.x * 0.75,
        sqrt(max(0.05, 1.0 - 0.5625 * (1.0 - sun_y * sun_y))),
        frame.light_direction.z * 0.75,
    ));
    let refr_offset = refr_sun.xz / max(0.32, refr_sun.y);

    // Use Horizon's real water surface plane when above the camera; fall back smoothly for deep/cave pools.
    let has_plane = frame.water_plane > frame.camera_position.y - 0.25
        && frame.water_plane < frame.camera_position.y + 80.0;
    let surface_y = select(max(62.88, frame.camera_position.y + 0.5), frame.water_plane, has_plane);
    let cam_depth = max(0.15, surface_y - frame.camera_position.y);
    let wind_t = frame.time * 0.48 * mix(0.82, 1.28, weather_wind());

    if neural_water {
        // Hybrid Multi-Turbidity Jerlov Neural Radiative Transfer + 8-step 3D Lagrangian
        // volumetric caustic curtain & shadow shaft modulator.
        let transmittance = exp(-extinction * travel);
        let eye_depth = cam_depth + (1.0 - access) * 10.0;
        let cos_vl = clamp(dot(ray, refr_sun), -1.0, 1.0);
        let neural_rad = neural_underwater_scatter(
            travel, eye_depth, ray.y, cos_vl, sun_y, jerlov.turbidity, jerlov.cdom
        );

        // 8-step jittered 3D volumetric caustic curtains + shadow raymarch.
        const SHAFT_STEPS: u32 = 8u;
        let step_s = travel / f32(SHAFT_STEPS);
        var shaft_sum = 0.0;
        var w_sum = 0.0;
        let curtain_gain = select(0.0, wave_strength * (1.0 - 0.65 * frame.rain), water_caustics);
        for (var i = 0u; i < SHAFT_STEPS; i += 1u) {
            let along = (f32(i) + jitter) * step_s;
            let point = ray * along;
            let world = frame.camera_position + point;
            let col_depth = max(0.0, surface_y - world.y) + (1.0 - access) * 10.0;
            var vis = 1.0;
            if pack_shadow_active() && access > 0.01 {
                let shadow = pack_shadow_position(point);
                if all(shadow >= vec3(0.0)) && all(shadow <= vec3(1.0)) {
                    vis = pack_shadow_compare(shadow.xy, shadow.z, false);
                    if frame.has_shadow == 0u {
                        vis = mix(vis, 1.0, pack_shadow_edge_fade(point, shadow));
                    }
                }
            }
            var shafts = 1.0;
            if curtain_gain > 0.005 {
                let surf_xz = world.xz + refr_offset * col_depth;
                let beam = underwater_caustic_curtain(surf_xz, wind_t);
                let depth_fade = exp(-0.075 * col_depth);
                shafts = 1.0 + curtain_gain * depth_fade * (beam * 2.15 - 0.30);
            }
            let w = exp(-0.085 * along);
            shaft_sum += vis * shafts * w;
            w_sum += w;
        }
        let mod_factor = shaft_sum / max(w_sum, 1e-4);
        // Subtle extra Petzold forward-scatter glare inside direct sunlit curtains.
        let forward_glare = 1.0 + 0.35 * pow(max(cos_vl, 0.0), 6.0) * smoothstep(0.35, 1.15, mod_factor);
        let direct_l = direct_atmosphere_light() * (cloud_shadow(vec3(0.0)) * access);
        let amb_l = photon_ambient_irradiance() * access;
        let sun_scatter = (direct_l * (mod_factor * forward_glare) + amb_l * 0.32) * neural_rad * 2.15;
        let fill_scatter = water_fill * (scattering_rgb * (vec3(1.0) - transmittance) / extinction);
        return AirFog(sun_scatter + fill_scatter, transmittance);
    }

    let steps = u32(clamp(8.0 + travel * 0.24, 8.0, 20.0));
    let step_size = travel / f32(steps);
    let transmission = exp(-extinction * step_size);
    let integral = scattering_rgb * (1.0 - transmission) / extinction;
    let cos_vl = clamp(dot(ray, refr_sun), -1.0, 1.0);
    let petzold = 0.45 * photon_phase(cos_vl, 0.84) + 0.55 * photon_phase(cos_vl, 0.35);
    let direct = direct_atmosphere_light() * (cloud_shadow(vec3(0.0)) * access) * petzold;
    let ambient = photon_ambient_irradiance() * PHOTON_ISOTROPIC * access;
    let curtain_gain = select(0.0, wave_strength * (1.0 - 0.65 * frame.rain), water_caustics);
    var result = AirFog(vec3(0.0), vec3(1.0));
    for (var i = 0u; i < steps; i += 1u) {
        let point = ray * ((f32(i) + jitter) * step_size);
        let world = frame.camera_position + point;
        let col_depth = max(0.0, surface_y - world.y) + (1.0 - access) * 10.0;
        let light_through_water = exp(-extinction * (col_depth / max(0.32, refr_sun.y)));
        var visibility = 1.0;
        if pack_shadow_active() && access > 0.01 {
            let shadow = pack_shadow_position(point);
            if all(shadow >= vec3(0.0)) && all(shadow <= vec3(1.0)) {
                visibility = pack_shadow_compare(shadow.xy, shadow.z, false);
                if frame.has_shadow == 0u {
                    visibility = mix(visibility, 1.0, pack_shadow_edge_fade(point, shadow));
                }
            }
        }
        var shafts = 1.0;
        if curtain_gain > 0.005 {
            let surf_xz = world.xz + refr_offset * col_depth;
            let beam = underwater_caustic_curtain(surf_xz, wind_t);
            shafts = 1.0 + curtain_gain * exp(-0.075 * col_depth) * (beam * 2.15 - 0.30);
        }
        let light = water_fill + ambient * sqrt(light_through_water)
            + direct * light_through_water * (visibility * shafts);
        result.scattering += result.transmittance * integral * light;
        result.transmittance *= transmission;
    }
    return result;
}
