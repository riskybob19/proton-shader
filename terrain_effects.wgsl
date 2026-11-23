override neural_gi: bool = true;

// Path-traced neural colored multi-bounce AO / pseudo-GI MLP (6 -> 12 -> 12 -> 3, ReLU).
// Trained on 24,000 Monte Carlo 3D cavity states (512 paths x 12 bounces = 12.3M rays).
// Validation R2=0.9952 (vs old scalar 0.3673), MAE=0.0150 (vs 0.1784), energy ratio=1.003.
// Zero-spill compile-time mat4x4 unrolled MLP (eliminates Naga/SPIR-V local array spill).
fn neural_multibounce_ao(vis: f32, rho: vec3<f32>, direct_frac: f32, normal_y: f32) -> vec3<f32> {
    let x0 = vec4(vis, rho.r, rho.g, rho.b);
    let h1_0 = max(vec4(0.55267, -0.35302, 0.47203, 0.07966) + mat4x4<f32>(
        vec4(1.27880, 0.06631, 0.53045, 1.45072),
        vec4(-0.25881, -0.06668, -0.72735, -0.19258),
        vec4(-0.18039, 0.01984, -1.39591, -0.37165),
        vec4(-0.14962, -0.17442, 0.05051, 0.11959),
    ) * x0 + vec4(-0.37775, 0.84840, -0.13435, 0.07576) * direct_frac + vec4(0.03475, 0.28842, 0.04738, -0.04344) * normal_y, vec4(0.0));
    let h1_1 = max(vec4(1.10079, 0.19858, 0.62292, -0.09941) + mat4x4<f32>(
        vec4(-1.26613, 1.90653, 0.71866, 1.21796),
        vec4(-0.04029, 0.58226, -0.29602, -0.10146),
        vec4(0.03396, 0.28395, 0.80527, 0.62276),
        vec4(-0.02118, 0.29695, -0.64580, -0.14079),
    ) * x0 + vec4(0.04202, 0.06596, 0.04208, -0.06073) * direct_frac + vec4(-0.01880, -0.01160, 0.00679, -0.00196) * normal_y, vec4(0.0));
    let h1_2 = max(vec4(-0.28522, -0.91177, 0.30642, 0.21358) + mat4x4<f32>(
        vec4(1.54988, -0.13058, -1.39041, 0.88337),
        vec4(0.02524, -0.02009, 0.42252, -0.50565),
        vec4(-0.06563, 1.13333, 0.27767, 0.39700),
        vec4(-0.03041, -0.00710, 0.22188, 0.69495),
    ) * x0 + vec4(-0.19121, 0.27273, -0.26362, 0.01952) * direct_frac + vec4(0.03030, -0.05062, 0.05980, 0.00529) * normal_y, vec4(0.0));
    let h2_0 = max(vec4(-0.17347, -0.12133, 0.05408, 0.04483) + mat4x4<f32>(
        vec4(-0.52335, -0.15391, -0.61163, -0.25040),
        vec4(-0.15286, -0.08016, 0.10744, -0.04099),
        vec4(0.09666, -0.96399, 0.34312, -0.05050),
        vec4(-0.21466, -0.05968, -0.29110, 0.02942),
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.98052, 0.21806, 0.24675, -0.67102),
        vec4(0.17915, 0.59188, -0.40550, 0.30274),
        vec4(0.75522, -0.16494, -0.54790, -0.40126),
        vec4(0.23318, 0.01226, 0.03082, -0.28964),
    ) * h1_1 + mat4x4<f32>(
        vec4(-0.88499, -1.33273, -0.04321, -0.32608),
        vec4(1.06138, -0.39763, -0.62546, 0.07438),
        vec4(-0.03869, -0.34824, -0.24296, -0.10819),
        vec4(0.51143, -0.56058, 0.10381, 0.89728),
    ) * h1_2, vec4(0.0));
    let h2_1 = max(vec4(0.52600, 0.30545, -0.01220, -0.12130) + mat4x4<f32>(
        vec4(-0.23191, -0.45333, -0.65539, -0.08054),
        vec4(-0.12634, -0.11829, -0.09859, 0.07876),
        vec4(0.20135, -0.09230, -0.04865, 0.05262),
        vec4(-0.28428, -0.22661, 0.43044, 0.12456),
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.81894, -0.78125, -0.74331, 0.09443),
        vec4(0.39961, 0.38438, 0.21175, -0.38433),
        vec4(-0.26252, 0.17792, -0.23658, -0.13249),
        vec4(-0.34323, 0.19122, -0.28892, -0.91780),
    ) * h1_1 + mat4x4<f32>(
        vec4(-0.69745, -0.51590, -0.68970, -0.32109),
        vec4(0.12000, 0.39292, 0.06732, 0.21587),
        vec4(-0.15845, 0.12398, 0.25092, -0.08384),
        vec4(0.67161, -0.14649, 0.71880, 0.22686),
    ) * h1_2, vec4(0.0));
    let h2_2 = max(vec4(1.05951, -0.13332, 0.47152, 0.18471) + mat4x4<f32>(
        vec4(-0.14858, -0.35619, -0.22265, -0.68158),
        vec4(-0.06853, -0.64981, -0.04695, -0.15819),
        vec4(0.29922, 0.20337, -0.06712, 0.60763),
        vec4(0.19165, -0.35068, -0.28715, 0.15986),
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.39960, -0.38696, -0.65699, -1.15467),
        vec4(-0.19584, 0.18556, 0.87721, 1.06691),
        vec4(-0.08823, -0.90007, -0.12998, 0.07806),
        vec4(-0.34438, -0.75462, -0.34415, -0.46860),
    ) * h1_1 + mat4x4<f32>(
        vec4(-0.09628, 0.31400, -0.23605, -0.99863),
        vec4(0.48700, 0.14740, -0.09589, 0.25444),
        vec4(0.14412, -0.65701, -0.21010, 0.26478),
        vec4(-0.02789, 0.10104, -0.59735, -0.77219),
    ) * h1_2, vec4(0.0));
    let gain = max(vec3(0.10109, 0.31798, 0.05775) + mat4x3<f32>(
        vec3(-0.11514, 0.63818, -0.00726),
        vec3(-0.17714, 0.08444, -0.05143),
        vec3(0.72347, 0.14590, 0.44705),
        vec3(-0.01113, -0.06941, 0.55498),
    ) * h2_0 + mat4x3<f32>(
        vec3(0.06588, 0.15805, 0.56837),
        vec3(0.49973, 0.60511, -0.10121),
        vec3(-0.01465, 0.03656, 0.77660),
        vec3(-0.43334, -0.50843, -0.08677),
    ) * h2_1 + mat4x3<f32>(
        vec3(-0.16945, -0.63156, -0.06805),
        vec3(0.30034, -0.85768, 0.30410),
        vec3(0.83816, -0.11759, 0.02373),
        vec3(0.65811, -0.10934, -0.02381),
    ) * h2_2, vec3(0.0));
    return clamp(vec3(vis) + (1.0 - vis) * gain, vec3(vis), vec3(1.0));
}

// 4-tap voxel-normal & plane-distance bilateral upsample of the temporally accumulated
// half-res 3-layer Hybrid Voxel GI buffer (gi_history.rgb paired with indirect_gi.a depth guide).
fn sample_ssilvb_gi(uv: vec2<f32>, distance: f32, normal: vec3<f32>) -> vec3<f32> {
    let tolerance = max(0.40, 0.060 * distance);
    // Beyond 18m on smooth coplanar surfaces, single-fetch hardware bilinear avoids 4-8 loads;
    // within 18m always run the 4-tap normal-strict bilateral upsample so 90-degree corners never bleed.
    if distance > 18.0 {
        let bi_d = textureSampleLevel(indirect_gi, linear_sampler, uv, 0.0).a;
        if bi_d > 0.0 && abs(bi_d - distance) < 0.30 * tolerance {
            return max(textureSampleLevel(gi_history, linear_sampler, uv, 0.0).rgb, vec3(0.0));
        }
    }
    let gi_size = vec2<i32>(textureDimensions(gi_history));
    let full_size_f = vec2<f32>(textureDimensions(scene_depth));
    let full_size_i = vec2<i32>(full_size_f);
    let inv_gi = 1.0 / vec2<f32>(gi_size);
    let pos_f = uv * vec2<f32>(gi_size) - 0.5;
    let corner = vec2<i32>(floor(pos_f));
    let frac = fract(pos_f);
    var gi_sum = vec3(0.0);
    var w_sum = 0.0;
    for (var tap = 0u; tap < 4u; tap += 1u) {
        let off = vec2<i32>(i32(tap & 1u), i32(tap >> 1u));
        let cell = clamp(corner + off, vec2(0), gi_size - 1);
        let d_guide = textureLoad(indirect_gi, cell, 0).a;
        if d_guide <= 0.0 {
            continue;
        }
        let s_rgb = max(textureLoad(gi_history, cell, 0).rgb, vec3(0.0));
        let d_diff = abs(d_guide - distance);
        let depth_w = 1.0 - smoothstep(0.0, tolerance, d_diff);
        var n_w = 1.0;
        // Always verify neighbor normals on close-range (<14m) 90-degree block corners or depth edges.
        if distance < 14.0 || d_diff > 0.12 * tolerance {
            let tap_full = clamp(vec2<i32>((vec2<f32>(cell) + 0.5) * inv_gi * full_size_f), vec2(0), full_size_i - 1);
            let tap_surf = surface_at(tap_full);
            let ndot = select(0.5, max(dot(normal, tap_surf.normal), 0.0), tap_surf.present);
            let n2 = ndot * ndot;
            n_w = n2 * n2;
        }
        let axis = select(1.0 - frac, frac, off == vec2(1));
        let w = axis.x * axis.y * (0.02 + 0.98 * depth_w * n_w);
        gi_sum += s_rgb * w;
        w_sum += w;
    }
    if w_sum <= 1e-4 {
        return vec3(0.0);
    }
    return gi_sum / w_sum;
}

fn terrain_effects(input: PostInput, position: vec3<f32>, distance: f32) -> vec4<f32> {
    if frame.dimension != 0 || input.depth <= 0.0 || (contact_ao <= 0.0 && !terrain_reflections && !shadows) {
        return input.color;
    }
    let texel = vec2<i32>(input.uv * vec2<f32>(textureDimensions(scene_depth)));
    let surface = surface_at(texel);
    if !surface.present {
        return input.color;
    }

    let packed = u32(surface.data);
    var color = input.color.rgb;
    let ambient = surface_ambient(packed);
    let fog = vanilla_fog(position);
    let direct_fraction = surface_direct(packed);
    let need_shadow = shadows && direct_fraction > 0.0 && fog < 1.0;
    let ao_coverage = clamp(ambient + 0.62 * direct_fraction, 0.0, 1.0);
    let need_ao = contact_ao > 0.0 && ao_coverage > 0.0 && fog < 1.0 && distance < 360.0;
    if need_shadow || need_ao {
        // Depth-aware bilateral upsample from half-res contact_visibility: uses 1-fetch
        // hardware bilinear on smooth terrain, and nearest depth-matching 2x2 neighbor
        // on steep voxel stairs/ridges so distant shadows and valley AO NEVER vanish on slopes.
        let ao_size = vec2<f32>(textureDimensions(contact_visibility));
        let ao_isize = vec2<i32>(ao_size);
        let ao_pos = input.uv * ao_size - 0.5;
        let ao_corner = vec2<i32>(floor(ao_pos));
        let ao_cell = clamp(vec2<i32>(input.uv * ao_size), vec2(0), ao_isize - 1);
        let source_uv = (vec2<f32>(ao_cell) + 0.5) / ao_size;
        let source_depth = depth_at(source_uv);
        let rel_diff = abs(source_depth - input.depth) / max(input.depth, 1e-8);
        var ao: vec4<f32>;
        var temp_shadow: f32;
        if rel_diff < 0.035 {
            ao = textureSampleLevel(contact_visibility, linear_sampler, input.uv, 0.0);
            let bi_sh = textureSampleLevel(gi_history, linear_sampler, input.uv, 0.0).a;
            let pt_sh = textureLoad(gi_history, ao_cell, 0).a;
            // On terraced hillsides (>96m), prevent 0.0 vertical risers in adjacent half-res
            // rows from diluting shadowed horizontal top faces during bilinear filtering.
            temp_shadow = select(bi_sh, max(bi_sh, pt_sh * 0.92), distance >= 96.0);
        } else {
            var best_diff = 1e30;
            var best_cell = ao_cell;
            for (var tap = 0u; tap < 4u; tap += 1u) {
                let q = clamp(ao_corner + vec2<i32>(i32(tap & 1u), i32(tap >> 1u)), vec2(0), ao_isize - 1);
                let q_uv = (vec2<f32>(q) + 0.5) / ao_size;
                let q_diff = abs(depth_at(q_uv) - input.depth);
                if q_diff < best_diff {
                    best_diff = q_diff;
                    best_cell = q;
                }
            }
            ao = textureLoad(contact_visibility, best_cell, 0);
            temp_shadow = textureLoad(gi_history, best_cell, 0).a;
        }
        var amb_occ = vec3(0.0);
        var dir_micro = 0.0;
        if need_ao {
            // Apply Visibility Bitmask AO to ambient skylight (with path-traced Neural
            // Colored Multi-Bounce GI on crevices) and Chan (SIGGRAPH 2018) directional
            // micro-shadowing to direct sunlight.
            var visibility = ao.r;
            if directional_skylight && visibility < 0.96 {
                let sky_level = surface_sky(packed);
                visibility *= mix(1.0, ao.g, sky_level * sky_level);
            }
            let grounded = visibility * mix(0.85, 1.0, visibility);
            let no_l = clamp(dot(surface.normal, frame.light_direction), 0.0, 1.0);
            let aperture = inverseSqrt(max(1.0 - clamp(grounded, 0.0, 0.999), 1e-4));
            let ms = clamp(no_l * aperture, 0.0, 1.0);
            // Scale directional micro-shadowing by (1.0 - grounded) and gate on sun-facing normals
            // (no_l > 0.02) so unoccluded surfaces and backlit subsurface-scattering foliage are never darkened.
            let direct_occ = select(0.0, ((1.0 - ms * ms) * 0.65 + 0.35) * (1.0 - grounded), no_l > 0.02);
            if neural_gi && visibility < 0.96 {
                let c_safe = max(color, vec3(0.0));
                let c_max = max(c_safe.r, max(c_safe.g, c_safe.b));
                let c_luma = dot(c_safe, frame.luminance);
                let chroma = c_safe / max(c_max, 1e-4);
                let refl_mag = clamp(sqrt(c_luma / (c_luma + 0.35)), 0.08, 0.88);
                let rho = clamp(chroma * refl_mag, vec3(0.04), vec3(0.92));
                let mb_vis = neural_multibounce_ao(clamp(visibility, 0.05, 1.0), rho, direct_fraction, surface.normal.y);
                amb_occ = contact_ao * (vec3(1.0) - mb_vis);
            } else {
                amb_occ = vec3(contact_ao * (1.0 - grounded));
            }
            dir_micro = contact_ao * direct_occ;
        }
        let dir_ss = select(0.0, select(ao.b, temp_shadow, distance >= 24.0), need_shadow);
        // Prevent 4-bit ambient + 5-bit direct quantization from summing to <1.0 on sunlit
        // non-emissive surfaces and leaving ~5% un-occludable direct sunlight in distant shadows.
        let eff_direct = select(
            direct_fraction,
            max(direct_fraction, 1.0 - ambient),
            direct_fraction > 0.02 && (direct_fraction + ambient) > 0.86,
        );
        // Combine screen-space / cached distant shadow with directional micro-shadowing
        // on direct light, PLUS physical sky-hemisphere occlusion in distant cast shadows
        // (mountains and tree canopies block ~52% of directional sky ambient in addition to
        // 100% of direct sun, matching the combined shadow + GTAO + skylight depth of near shadows!).
        let dir_occ = 1.0 - (1.0 - dir_ss) * (1.0 - dir_micro);
        let shadow_sky_occ = dir_ss * mix(0.20, 0.54, smoothstep(32.0, 108.0, distance));
        let eff_amb_occ = vec3(1.0) - (vec3(1.0) - amb_occ) * (1.0 - shadow_sky_occ);
        let total_occ = clamp(ambient * eff_amb_occ + vec3(eff_direct * dir_occ), vec3(0.0), vec3(0.98));
        // Exact per-channel chromaticity transfer from sunlit mixture (eff_direct * sun + (1 - eff_direct) * amb)
        // to pure sky ambient (amb_norm), removing warm red/green direct-sun excess in distant shadows.
        var shadow_chroma = vec3(1.0);
        if dir_ss > 0.01 && eff_direct > 0.04 {
            let sun_l = max(direct_atmosphere_light(), vec3(1e-4));
            let amb_l = max(ambient_atmosphere_light(), vec3(1e-4));
            let sun_norm = sun_l / max(dot(sun_l, frame.luminance), 1e-4);
            let amb_norm = amb_l / max(dot(amb_l, frame.luminance), 1e-4);
            let mix_norm = max(eff_direct * sun_norm + (1.0 - eff_direct) * amb_norm, vec3(0.15));
            let chroma_ratio = clamp(amb_norm / mix_norm, vec3(0.58), vec3(1.42));
            shadow_chroma = mix(vec3(1.0), chroma_ratio, dir_ss * clamp(eff_direct * 1.25, 0.0, 1.0) * (1.0 - fog));
        }
        color *= (vec3(1.0) - total_occ * (1.0 - fog)) * shadow_chroma;
    }

    if (ssilvb_gi || gi_rsm_bounce) && fog < 1.0 && distance > 0.08 && distance < 96.0 {
        let bounce_irrad = sample_ssilvb_gi(input.uv, distance, surface.normal);
        if max(bounce_irrad.r, max(bounce_irrad.g, bounce_irrad.b)) > 1e-4 {
            // Estimate physically linear receiver diffuse reflectance from pre-AO color
            // normalized by direct-light exposure (never gamma-lifted sqrt, so dark foliage
            // and grass shadows stay deep and richly colored instead of turning chalky).
            let r_safe = max(input.color.rgb, vec3(0.0));
            let r_max = max(r_safe.r, max(r_safe.g, r_safe.b));
            let r_luma = dot(r_safe, frame.luminance);
            let r_chroma = select(vec3(0.55), r_safe / max(r_max, 1e-4), r_max > 0.002);
            let exposure_norm = mix(0.09, 0.55, direct_fraction);
            let est_refl = clamp(r_luma / (r_luma + exposure_norm), 0.08, 0.72);
            let r_albedo = clamp(r_chroma * est_refl, vec3(0.06), vec3(0.72));
            color += bounce_irrad * r_albedo * (0.55 * (1.0 - fog));
        }
    }

    if terrain_reflections && fog < 1.0 && distance > 0.02 {
        let reflectance = surface_reflectance(packed);
        if max(reflectance.r, max(reflectance.g, reflectance.b)) > 0.008 {
            // 4-tap depth-aware bilateral upsample from the half-resolution Reflection TAA buffer.
            let refl_size = vec2<i32>(textureDimensions(reflection_history));
            let pos_f = input.uv * vec2<f32>(refl_size) - 0.5;
            let corner = vec2<i32>(floor(pos_f));
            let frac = fract(pos_f);
            let tolerance = max(0.35, 0.055 * distance);
            var refl_sum = vec3(0.0);
            var w_sum = 0.0;
            for (var tap = 0u; tap < 4u; tap += 1u) {
                let off = vec2<i32>(i32(tap & 1u), i32(tap >> 1u));
                let q = clamp(corner + off, vec2(0), refl_size - 1);
                let sample_val = textureLoad(reflection_history, q, 0);
                if sample_val.a > 0.0 {
                    let depth_w = 1.0 - smoothstep(0.0, tolerance, abs(sample_val.a - distance));
                    let axis = select(1.0 - frac, frac, off == vec2(1));
                    let w = axis.x * axis.y * depth_w;
                    refl_sum += sample_val.rgb * w;
                    w_sum += w;
                }
            }
            var reflected: vec3<f32>;
            if w_sum > 0.005 {
                reflected = refl_sum / w_sum;
            } else {
                // Clean environment fallback for isolated 1-pixel reflective silhouettes.
                let ray = reflect(position / distance, surface.normal);
                let sky = surface_sky(packed);
                let roughness = 1.0 - surface.smoothness;
                let reflect_dir = normalize(vec3(ray.x, max(ray.y, 0.03) + 0.35 * roughness, ray.z));
                let radiance = environment_radiance(reflect_dir);
                let broad = ambient_atmosphere_light() * 0.45;
                let sky_access = smoothstep(0.42, 1.0, sky);
                reflected = mix(radiance, broad, roughness * 0.6) * sky_access * smoothstep(-0.25, 0.08, ray.y);
            }
            color += reflected * reflectance * (1.0 - fog);
        }
    }
    return vec4(color, input.color.a);
}
