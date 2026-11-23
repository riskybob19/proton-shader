// Low-Cost 3-Layer Hybrid Voxel Global Illumination (Horizon API 0.2, +0.00 MiB fixed VRAM):
//   Layer 1: EGSR 2023 SSILVB (32-bit cosine-weighted visibility bitmask + taa_history_previous multi-bounce)
//   Layer 2: Bitmask Bent-Normal Directional Colored Sky GI (samples skylight_diffuse in full RGB)
//   Layer 3: Off-Screen World-Space Shadow-Map Bounce (Depth-RSM over 256x256m shadow_map, 360 deg)

fn gi_sky_irradiance(normal: vec3<f32>) -> vec3<f32> {
    let plane = normal.xz / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    let oct = 0.5 + 0.5 * select(plane, octahedral_fold(plane), normal.y < 0.0);
    let uv = skylight_atlas_uv(oct, vec2<f32>(textureDimensions(cloud_aux)));
    return max(textureSampleLevel(cloud_aux, scattering_atlas_sampler, uv, 0.0).rgb, vec3(0.0));
}

fn gi_cloud_vis(position: vec3<f32>) -> f32 {
    if frame.light_direction.y <= 0.05 {
        return 1.0;
    }
    let projected = position.xz - frame.light_direction.xz * (position.y / frame.light_direction.y);
    let uv = 0.5 + projected * (1.0 / CLOUD_SHADOW_EXTENT);
    if any(uv <= vec2(0.01)) || any(uv >= vec2(0.99)) {
        return 1.0;
    }
    let cell = vec2(0.5) + uv * vec2(CLOUD_MAP_WIDTH - 1.0, CLOUD_MAP_HEIGHT - 1.0)
        + vec2(0.0, ENV_HEIGHT);
    return clamp(textureSampleLevel(cloud_aux, scattering_atlas_sampler,
        cell / vec2<f32>(textureDimensions(cloud_aux)), 0.0).r, 0.15, 1.0);
}

// Reproject sender into previous frame's TAA history for free multi-bounce feedback
// (bounce N in taa_history_previous feeds directly into bounce N+1).
fn ssilvb_sample_radiance(s_pos: vec3<f32>, cell: vec2<i32>, size_f: vec2<f32>, size_i: vec2<i32>, use_history: bool) -> vec3<f32> {
    var rad = max(textureLoad(scene_color, cell, 0).rgb, vec3(0.0));
    if use_history && max(rad.r, max(rad.g, rad.b)) > 0.015 && frame.delta_time > 0.0 && frame.delta_time < 0.16 {
        let old_pos = s_pos + frame.camera_motion;
        let clip = frame.previous_view_projection * vec4(old_pos, 1.0);
        if clip.w > 0.05 {
            let ndc = clip.xy / clip.w;
            if all(abs(ndc) < vec2(0.995)) {
                let prev_uv = 0.5 + ndc * vec2(0.5, -0.5);
                let prev_cell = clamp(vec2<i32>(prev_uv * size_f), vec2(0), size_i - 1);
                let hist = textureLoad(taa_history_previous, prev_cell, 0);
                let expected = length(old_pos);
                if hist.a > 0.0 && abs(hist.a - expected) < max(0.65, 0.12 * expected) {
                    rad = mix(rad, max(hist.rgb, vec3(0.0)), 0.48);
                }
            }
        }
    }
    // Energy-safe soft firefly knee prevents single specular pixels from injecting GI noise.
    let luma = dot(rad, frame.luminance);
    return rad * (1.0 / (1.0 + 0.10 * luma));
}

struct SsilvbSliceResult {
    irradiance: vec3<f32>,
    bitmask: u32,
}

// March one direction along a horizon slice, updating the 32-bit cosine-weighted
// visibility bitmask and integrating newly visible solid angle from lit senders.
fn ssilvb_march_dir(center: vec3<f32>, toward_eye: vec3<f32>, tangent: vec3<f32>,
    normal: vec3<f32>, cos_gamma: f32, sin_gamma: f32, uv: vec2<f32>, step_uv: vec2<f32>,
    size_f: vec2<f32>, size_i: vec2<i32>, radius: f32, thickness: f32,
    jitter: f32, steps: u32, init_mask: u32) -> SsilvbSliceResult {
    let step_pixels = length(step_uv * size_f);
    if step_pixels < 0.25 {
        return SsilvbSliceResult(vec3(0.0), init_mask);
    }
    var bitmask = init_mask;
    var accum = vec3(0.0);
    let f_steps = f32(steps);
    let inv_steps = 1.0 / f_steps;
    let base_off = 1.5 / step_pixels;
    let radius2 = radius * radius;
    let inv_size = 1.0 / size_f;

    for (var tap = 0u; tap < steps; tap += 1u) {
        let t = (f32(tap) + jitter) * inv_steps;
        let sample_uv = uv + step_uv * (t * t * f_steps + base_off);
        if any(sample_uv <= vec2(0.002)) || any(sample_uv >= vec2(0.998)) {
            break;
        }
        let cell = clamp(vec2<i32>(sample_uv * size_f), vec2(0), size_i - 1);
        let depth = textureLoad(scene_depth, cell, 0);
        if depth <= 0.0 {
            continue;
        }
        let sample_center = (vec2<f32>(cell) + 0.5) * inv_size;
        let s_pos = view_position(sample_center, depth);
        let separation = s_pos - center;
        let squared = dot(separation, separation);
        if squared <= 0.04 || squared >= radius2 {
            continue;
        }
        let height = dot(normal, separation);
        if height <= 0.025 || height * height <= 0.0012 * squared {
            continue;
        }
        let inv_dist = inverseSqrt(squared);
        let dist = squared * inv_dist;
        let dist_falloff = 1.0 - smoothstep(0.62 * radius, radius, dist);
        let sample_thick = max(thickness, dist * 0.48);

        let s_t = dot(separation, tangent);
        let s_v = dot(separation, toward_eye);
        let sn_f = s_v * cos_gamma + s_t * sin_gamma;
        let sp_f = s_t * cos_gamma - s_v * sin_gamma;
        let sin_f = select(select(-1.0, 1.0, sp_f >= 0.0),
            sp_f * inverseSqrt(max(sn_f * sn_f + sp_f * sp_f, 1e-12)), sn_f > 0.0);

        let bv = s_v - sample_thick;
        let sn_b = bv * cos_gamma + s_t * sin_gamma;
        let sp_b = s_t * cos_gamma - bv * sin_gamma;
        let sin_b = select(select(-1.0, 1.0, sp_b >= 0.0),
            sp_b * inverseSqrt(max(sn_b * sn_b + sp_b * sp_b, 1e-12)), sn_b > 0.0);

        let u_f = 0.5 + 0.5 * sin_f;
        let u_b = 0.5 + 0.5 * sin_b;
        let lo = min(u_f, u_b);
        let hi = max(u_f, u_b);
        let span = (hi - lo) * dist_falloff;
        if span <= 0.004 {
            continue;
        }
        let mid = 0.5 * (lo + hi);
        let a = clamp(mid - 0.5 * span, 0.0, 1.0);
        let b = clamp(mid + 0.5 * span, 0.0, 1.0);
        let start_bit = u32(clamp(floor(a * 32.0), 0.0, 31.0));
        let bit_count = u32(clamp(ceil((b - a) * 32.0), 0.0, 32.0 - f32(start_bit)));
        if bit_count == 0u {
            continue;
        }
        let sector_mask = select(((1u << bit_count) - 1u) << start_bit, 0xffffffffu, bit_count >= 32u);
        let new_bits = sector_mask & (~bitmask);
        bitmask |= sector_mask;
        if new_bits == 0u {
            continue;
        }

        let sender = surface_at(cell);
        if !sender.present {
            continue;
        }
        let dir = separation * inv_dist;
        let sender_facing = smoothstep(-0.15, 0.55, dot(sender.normal, -dir));
        if sender_facing <= 0.01 {
            continue;
        }
        let sector_weight = f32(countOneBits(new_bits)) * (1.0 / 32.0);
        let rad = ssilvb_sample_radiance(s_pos, cell, size_f, size_i, squared < 64.0);
        accum += rad * (sector_weight * sender_facing);
    }
    return SsilvbSliceResult(accum, bitmask);
}

// Layer 3: Off-Screen World-Space Shadow-Map Bounce (Depth-RSM).
// Samples 6 temporally rotated sunlit VPLs directly from Horizon's 1024x1024 shadow_map
// (covering 256x256 blocks around the player, 360 deg including behind the camera),
// reconstructing exact 3D world-space VPL offsets from orthographic frame.shadow_view_proj.
fn shadow_map_rsm_bounce(center: vec3<f32>, normal: vec3<f32>, distance: f32,
    sky_level: f32, world_to_clip: mat4x4<f32>, size_f: vec2<f32>, size_i: vec2<i32>,
    jitter: f32) -> vec3<f32> {
    if !gi_rsm_bounce || !pack_shadow_active() || frame.light_direction.y <= 0.045 || distance >= 80.0 {
        return vec3(0.0);
    }
    let sun_light = direct_atmosphere_light();
    if max(sun_light.r, max(sun_light.g, sun_light.b)) <= 0.005 {
        return vec3(0.0);
    }
    let p = pack_shadow_position(center);
    if any(p.xy <= vec2(0.02)) || any(p.xy >= vec2(0.98)) || p.z <= 0.0 || p.z >= 1.0 {
        return vec3(0.0);
    }
    let shadow_fade = (1.0 - pack_shadow_edge_fade(center, p)) * (1.0 - smoothstep(56.0, 80.0, distance));
    if shadow_fade <= 0.01 {
        return vec3(0.0);
    }

    // Orthographic shadow_view_proj row basis vectors for exact 3D VPL reconstruction.
    let sp = frame.shadow_view_proj;
    let rx = vec3(sp[0].x, sp[1].x, sp[2].x);
    let ry = vec3(sp[0].y, sp[1].y, sp[2].y);
    let rz = vec3(sp[0].z, sp[1].z, sp[2].z);
    let inv_rx2 = 1.0 / max(dot(rx, rx), 1e-8);
    let inv_ry2 = 1.0 / max(dot(ry, ry), 1e-8);
    let inv_rz2 = 1.0 / max(dot(rz, rz), 1e-8);

    let sh_size = pack_shadow_size();
    let sh_size_f = vec2<f32>(sh_size);
    // 1 shadow texel in world meters (~0.25m at 1024 / 256m) -> search radius ~20m.
    let radius_uv = clamp(20.0 * sqrt(dot(rx, rx)) * 0.5, 0.035, 0.14);

    let rot_a = jitter * TAU;
    let ca = cos(rot_a);
    let sa = sin(rot_a);
    let rot = mat2x2<f32>(vec2(ca, sa), vec2(-sa, ca));
    var flux_sum = 0.0;
    let check_dedup = ssilvb_gi && distance < 36.0;
    for (var k = 0u; k < 3u; k += 1u) {
        let v_off = select(select(vec2(0.408248, 0.0), vec2(-0.521402, 0.477640), k == 1u),
            vec2(0.079788, -0.909359), k == 2u);
        let duv = (rot * v_off) * radius_uv;
        let tap_uv = p.xy + duv;
        if any(tap_uv <= vec2(0.005)) || any(tap_uv >= vec2(0.995)) {
            continue;
        }
        let tap_px = clamp(vec2<i32>(tap_uv * sh_size_f), vec2(0), sh_size - 1);
        let tap_z = pack_shadow_depth(tap_px, false);
        if tap_z <= 0.0005 || tap_z >= 0.9995 {
            continue;
        }
        // Convert shadow-UV & depth delta to exact 3D world-space VPL offset.
        // shadow_uv.x = 0.5 + 0.5 * ndc.x, shadow_uv.y = 0.5 - 0.5 * ndc.y.
        let d_ndc_x = 2.0 * duv.x;
        let d_ndc_y = -2.0 * duv.y;
        let d_z = tap_z - p.z;
        let delta = rx * (d_ndc_x * inv_rx2) + ry * (d_ndc_y * inv_ry2) + rz * (d_z * inv_rz2);
        let d2 = dot(delta, delta);
        if d2 < 2.25 || d2 > 625.0 {
            continue;
        }
        let inv_d = inverseSqrt(d2);
        let dir = delta * inv_d;
        // Must lie in the receiver's upper hemisphere (avoid self-bounce on flat ground).
        let n_dot_dir = dot(normal, dir);
        if n_dot_dir <= 0.08 {
            continue;
        }
        // Sunlit VPL faces generally toward the sun; weight smoothly by bounce alignment.
        let vpl_align = clamp(0.22 + 0.78 * max(dot(frame.light_direction, -dir), 0.0), 0.0, 1.0);
        let dist = d2 * inv_d;
        let atten = (1.0 / (1.0 + 0.055 * d2)) * (1.0 - smoothstep(13.0, 23.0, dist));

        // Deduplicate against on-screen SSILVB: full weight when the sunlit VPL is off-screen
        // or occluded from the camera, light fill weight when SSILVB already samples it on-screen.
        var offscreen_w = 1.0;
        if check_dedup {
            let vpl_world = center + delta;
            let vpl_clip = world_to_clip * vec4(vpl_world, 1.0);
            if vpl_clip.w > 0.05 {
                let vpl_ndc = vpl_clip.xyz / vpl_clip.w;
                if all(abs(vpl_ndc.xy) < vec2(0.96)) && vpl_ndc.z > 0.0 && vpl_ndc.z < 1.0 {
                    let vpl_uv = 0.5 + vpl_ndc.xy * vec2(0.5, -0.5);
                    let vpl_cell = clamp(vec2<i32>(vpl_uv * size_f), vec2(0), size_i - 1);
                    let s_d = textureLoad(scene_depth, vpl_cell, 0);
                    if s_d > 0.0 && abs(s_d - vpl_ndc.z) < max(0.0025, 0.12 * vpl_ndc.z) {
                        offscreen_w = 0.14;
                    }
                }
            }
        }
        flux_sum += n_dot_dir * vpl_align * atten * offscreen_w;
    }

    if flux_sum <= 1e-4 {
        return vec3(0.0);
    }
    // Smooth climate-aware bounced ground albedo (golden-ochre deserts, lush green plains/forests,
    // bright ivory snowfields, neutral warm stone cliffs/caves) modulated by cloud shadows.
    let temp = clamp(frame.temperature, 0.0, 1.0);
    let rain_climate = clamp(frame.downfall, 0.0, 1.0);
    let snow_w = 1.0 - smoothstep(0.05, 0.25, temp);
    let desert_w = smoothstep(0.75, 1.0, temp) * (1.0 - smoothstep(0.10, 0.40, rain_climate));
    let lush_albedo = vec3(0.38, 0.46, 0.26);
    let desert_albedo = vec3(0.72, 0.60, 0.42);
    let snow_albedo = vec3(0.78, 0.82, 0.88);
    let cliff_albedo = vec3(0.45, 0.43, 0.41);
    let outdoor_albedo = mix(mix(lush_albedo, desert_albedo, desert_w), snow_albedo, snow_w);
    let ground_albedo = mix(cliff_albedo, outdoor_albedo, smoothstep(0.10, 0.70, sky_level));

    let cloud_vis = gi_cloud_vis(center);
    // Gate smoothly by sky access near cave mouths so sealed deep caves never leak sunlit RSM.
    let cave_gate = smoothstep(0.02, 0.28, sky_level + 0.14 * smoothstep(0.20, 0.85, frame.eye_light_smooth.y));
    // Convert sun irradiance * ground albedo to cosine-integrated radiance units (/ pi)
    // scaled by 2.0x for 3 temporally rotated Vogel taps (energy-identical to 6 taps).
    return sun_light * (ground_albedo * (1.0 / PHOTON_PI)) * (flux_sum * 0.52 * shadow_fade * cloud_vis * cave_gate);
}

fn post(input: PostInput) -> vec4<f32> {
    if (!ssilvb_gi && !gi_rsm_bounce) || frame.dimension != 0 || input.depth <= 0.0 {
        return vec4(0.0);
    }
    let size_f = vec2<f32>(textureDimensions(scene_depth));
    let size_i = vec2<i32>(size_f);
    let cell = clamp(vec2<i32>(input.uv * size_f), vec2(0), size_i - 1);
    let surface = surface_at(cell);
    let depth = textureLoad(scene_depth, cell, 0);
    if !surface.present || depth <= 0.0 {
        return vec4(0.0);
    }

    let uv = (vec2<f32>(cell) + 0.5) / size_f;
    let center = view_position(uv, depth);
    let distance = length(center);
    if distance < 0.08 || distance >= 96.0 {
        return vec4(0.0);
    }
    let dist_fade = 1.0 - smoothstep(66.0, 96.0, distance);
    let packed = u32(surface.data);
    let sky_level = surface_sky(packed);

    let world_to_clip = frame.projection * frame.view;
    let spatial_ign = fract(dot(floor(input.pixel), vec2(0.06711056, 0.00583715)) * 52.9829189);
    let jitter = fract(spatial_ign + f32(frame.frame_index & 31u) * 0.61803399);

    var total_gi = vec3(0.0);

    // Layer 1 + Layer 2: SSILVB Multi-Bounce GI + Bitmask Bent-Normal Colored Sky GI
    if ssilvb_gi {
        let toward_eye = -center / distance;
        let camera_right = normalize(frame.view_inverse[0].xyz);
        let camera_up = normalize(frame.view_inverse[1].xyz);
        let right = normalize(cross(camera_up, toward_eye));
        let up = cross(toward_eye, right);

        let near_detail = distance < 22.0;
        let slices = 1u;
        let steps = select(3u, 4u, near_detail);
        let radius = clamp(distance * 0.45, 6.0, 24.0);
        let thickness = clamp(distance * 0.065, 0.75, 4.2);
        let center_clip = world_to_clip * vec4(center, 1.0);
        let center_ndc = center_clip.xy / max(center_clip.w, 0.02);

        var gi_sum = vec3(0.0);
        var norm_sum = 0.0;
        var open_vis_sum = 0.0;
        var bent = vec3(0.0);

        for (var slice = 0u; slice < slices; slice += 1u) {
            let angle = (f32(slice) + jitter) * PHOTON_PI / f32(slices);
            let tangent = right * cos(angle) + up * sin(angle);
            let axis = cross(toward_eye, tangent);
            let projected_normal = surface.normal - axis * dot(surface.normal, axis);
            let normal_length = length(projected_normal);
            if normal_length <= 0.001 {
                continue;
            }
            let inv_norm = 1.0 / normal_length;
            let raw_cos = dot(projected_normal, toward_eye) * inv_norm;
            let raw_sin = dot(projected_normal, tangent) * inv_norm;
            let cos_gamma = max(raw_cos, 0.0);
            let sin_gamma = select(select(-1.0, 1.0, raw_sin >= 0.0), raw_sin, raw_cos >= 0.0);

            let offset = world_to_clip * vec4(tangent * (radius / f32(steps)), 0.0);
            let probe = center_clip + offset;
            var step_uv = (probe.xy / max(probe.w, 0.02) - center_ndc) * vec2(0.5, -0.5);
            let total_screen = length(step_uv) * f32(steps);
            if total_screen > 0.32 {
                step_uv *= 0.32 / total_screen;
            }
            let tap_jitter = 0.20 + 0.75 * fract(jitter + f32(slice) * 0.618034);

            let neg = ssilvb_march_dir(center, toward_eye, tangent, surface.normal, cos_gamma, sin_gamma,
                uv, -step_uv, size_f, size_i, radius, thickness, tap_jitter, steps, 0u);
            let pos = ssilvb_march_dir(center, toward_eye, tangent, surface.normal, cos_gamma, sin_gamma,
                uv, step_uv, size_f, size_i, radius, thickness, tap_jitter, steps, neg.bitmask);
            gi_sum += (neg.irradiance + pos.irradiance) * normal_length;
            norm_sum += normal_length;

            // Layer 2: Accumulate unoccluded bent normal directly from ~bitmask (zero extra rays).
            let occ_mask = pos.bitmask;
            let left_open = 16.0 - f32(countOneBits(occ_mask & 0x0000ffffu));
            let right_open = 16.0 - f32(countOneBits(occ_mask >> 16u));
            let slice_vis = (left_open + right_open) * (1.0 / 32.0);
            let weighted = slice_vis * normal_length;
            open_vis_sum += weighted;
            let delta_a = ((right_open - left_open) * (1.0 / 32.0)) * (0.5 * PHOTON_PI);
            let cd = cos(delta_a);
            let sd = sin(delta_a);
            let cos_med = cos_gamma * cd - sin_gamma * sd;
            let sin_med = sin_gamma * cd + cos_gamma * sd;
            bent += weighted * (toward_eye * cos_med + tangent * sin_med);
        }

        let inv_n = 1.0 / max(norm_sum, 1e-4);
        total_gi += (gi_sum * inv_n) * dist_fade;

        // Directional colored sky shift along the unoccluded bent normal (sunset gold vs cool sky).
        let open_vis = clamp(open_vis_sum * inv_n, 0.0, 1.0);
        let sky_w = sky_level * sky_level;
        if sky_w > 0.02 && open_vis < 0.94 && open_vis > 0.04 {
            let bent_dir = normalize(bent + 0.25 * surface.normal + 0.001 * camera_right);
            let bent_normal = normalize(mix(surface.normal, bent_dir, 1.0 - open_vis));
            let sky_bent = gi_sky_irradiance(bent_normal);
            let sky_base = gi_sky_irradiance(surface.normal);
            // Add strictly positive directional color lift in radiance units (/ pi) so
            // overhangs pick up warm sunset / cool sky contrast without washing out noon shadows.
            let sky_delta = max(sky_bent - sky_base, vec3(0.0));
            total_gi += sky_delta * (0.14 * sky_w * open_vis * dist_fade);
        }
    }

    // Layer 3: Off-Screen World-Space Shadow-Map Bounce (Depth-RSM, 360 deg around player).
    total_gi += shadow_map_rsm_bounce(center, surface.normal, distance, sky_level,
        world_to_clip, size_f, size_i, fract(jitter + 0.3719));

    return vec4(total_gi, distance);
}
