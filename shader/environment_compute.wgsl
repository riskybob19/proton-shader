override cloud_history: bool = true;

// 4x4 Bayer order (0..15): every 2x2 sub-block updates every 4 frames and the entire
// visible sky refreshes in 16 frames (eliminates 64-frame trailing lag and smear).
fn cloud_bayer4_index(p: vec2<u32>) -> u32 {
    let x = p.x & 3u;
    let y = p.y & 3u;
    let b0 = ((x & 1u) ^ (y & 1u)) | ((y & 1u) << 1u);
    let b1 = (((x >> 1u) & 1u) ^ ((y >> 1u) & 1u)) | (((y >> 1u) & 1u) << 1u);
    return (b0 << 2u) | b1;
}

// 5-tap Catmull-Rom history filter clamped to depth-valid 2x2 bounds:
// eliminates recursive bilinear diffusion smear during camera motion without ringing.
fn cloud_catmull_rom_history(raw_texel: vec2<f32>, c_min: vec4<f32>, c_max: vec4<f32>) -> vec4<f32> {
    let res = vec2(ENV_WIDTH, ENV_HEIGHT);
    let center = floor(raw_texel - 0.5) + 0.5;
    let f = raw_texel - center;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = f2 - 0.5 * (f3 + f);
    let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    let w3 = 0.5 * (f3 - f2);
    let w2 = 1.0 - w0 - w1 - w3;
    let w12 = w1 + w2;
    let uv_min = vec2(0.5) / res;
    let uv_max = (res - vec2(0.5)) / res;
    let t0 = clamp((center - 1.0) / res, uv_min, uv_max);
    let t12 = clamp((center + w2 / max(w12, vec2(1e-5))) / res, uv_min, uv_max);
    let t3 = clamp((center + 2.0) / res, uv_min, uv_max);
    let s0 = textureSampleLevel(cloud_environment_map_previous, scattering_atlas_sampler, vec2(t12.x, t0.y), 0.0);
    let s1 = textureSampleLevel(cloud_environment_map_previous, scattering_atlas_sampler, vec2(t0.x, t12.y), 0.0);
    let s2 = textureSampleLevel(cloud_environment_map_previous, scattering_atlas_sampler, t12, 0.0);
    let s3 = textureSampleLevel(cloud_environment_map_previous, scattering_atlas_sampler, vec2(t3.x, t12.y), 0.0);
    let s4 = textureSampleLevel(cloud_environment_map_previous, scattering_atlas_sampler, vec2(t12.x, t3.y), 0.0);
    let sum = s0 * (w12.x * w0.y) + s1 * (w0.x * w12.y) + s2 * (w12.x * w12.y)
        + s3 * (w3.x * w12.y) + s4 * (w12.x * w3.y);
    let norm = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return clamp(max(sum / max(norm, 1e-5), vec4(0.0)), c_min, c_max);
}

fn cloud_coverage(world: vec2<f32>) -> f32 {
    return cumulus_coverage(world);
}

fn nvdf_update_voxel(pixel: vec2<i32>, weather_key: f32) {
    let v = nvdf_pixel_to_voxel(pixel);
    let anchor = nvdf_snap_anchor(frame.camera_position.xz);
    let prev_meta = nvdf_read_meta(cloud_aux_previous);
    let prev_valid = prev_meta.w > 0.5 && frame.delta_time > 0.0 && frame.delta_time < 0.25;
    let shift_xz = select(vec2<i32>(999), vec2<i32>(round((anchor - prev_meta.xy) / NVDF_CELL_XZ)), prev_valid);
    let weather_steady = abs(weather_key - prev_meta.z) < 0.012;

    // 1/2 temporal checkerboard amortization when grid anchor and weather are steady (~32k voxels/frame).
    if prev_valid && shift_xz.x == 0 && shift_xz.y == 0 && weather_steady {
        if ((v.x ^ v.y ^ v.z) & 1) != i32(frame.frame_index & 1u) {
            textureStore(cloud_aux, pixel, textureLoad(cloud_aux_previous, pixel, 0));
            return;
        }
    }

    let world = nvdf_voxel_world_pos(v, anchor);
    let stratus = cumulus_stratus();
    let h_scale = cumulus_height_scale(stratus);
    let center_prof = nubis_dimensional_profile(world, world.y, cloud_coverage(world.xz), stratus, h_scale);

    // Conservative 5-tap sub-cell occupancy envelope prevents SDF skipping past thin <64m wisps.
    var dp_cons = center_prof.x;
    if dp_cons <= 0.002 && volumetric_clouds && cloud_cover > 0.0 {
        let off = 22.0;
        let d0 = nubis_dimensional_profile(world + vec3( off, 0.0,  off), world.y, cloud_coverage(world.xz + vec2( off,  off)), stratus, h_scale).x;
        let d1 = nubis_dimensional_profile(world + vec3(-off, 0.0,  off), world.y, cloud_coverage(world.xz + vec2(-off,  off)), stratus, h_scale).x;
        let d2 = nubis_dimensional_profile(world + vec3( off, 0.0, -off), world.y, cloud_coverage(world.xz + vec2( off, -off)), stratus, h_scale).x;
        let d3 = nubis_dimensional_profile(world + vec3(-off, 0.0, -off), world.y, cloud_coverage(world.xz + vec2(-off, -off)), stratus, h_scale).x;
        dp_cons = max(dp_cons, max(max(d0, d1), max(d2, d3)));
    }

    // Conservative 3D SDF propagation across 6 axial neighbors in previous frame's grid.
    var sdf = 0.0;
    if dp_cons <= 0.002 {
        if prev_valid {
            sdf = 256.0;
            let pv = v + vec3<i32>(shift_xz.x, 0, shift_xz.y);
            var offsets = array<vec3<i32>, 6>(
                vec3(1, 0, 0), vec3(-1, 0, 0),
                vec3(0, 0, 1), vec3(0, 0, -1),
                vec3(0, 1, 0), vec3(0, -1, 0),
            );
            var steps = array<f32, 6>(NVDF_CELL_XZ, NVDF_CELL_XZ, NVDF_CELL_XZ, NVDF_CELL_XZ, NVDF_CELL_Y, NVDF_CELL_Y);
            for (var n = 0u; n < 6u; n += 1u) {
                let nv = pv + offsets[n];
                if all(nv >= vec3(0)) && all(nv < vec3(64, 16, 64)) {
                    let nbr = textureLoad(cloud_aux_previous, nvdf_voxel_to_pixel(nv), 0);
                    let sd = steps[n];
                    sdf = min(sdf, select(nbr.b + sd, sd * 0.5, nbr.r > 0.002));
                }
            }
        } else {
            sdf = 32.0;
        }
    }

    // Amortized 5-tap solar optical depth (density-meters) baked directly into voxel
    // alpha (rgba16float avoids fp16 exp/log underflow so thick cloud bellies retain
    // full 15..35+ optical depth and deep 3D self-shadowing).
    var sun_depth = 0.0;
    let light = frame.light_direction;
    if volumetric_clouds && cloud_cover > 0.0 && light.y > 0.02 {
        var dists = array<f32, 5>(18.0, 44.0, 88.0, 155.0, 250.0);
        var segs = array<f32, 5>(22.0, 34.0, 56.0, 80.0, 110.0);
        for (var s = 0u; s < 5u; s += 1u) {
            let sp = world + light * dists[s];
            if sp.y >= CLOUD_BASE && sp.y <= CLOUD_TOP {
                var s_dp = 0.0;
                if prev_valid && nvdf_in_bounds(sp, prev_meta.xy) {
                    s_dp = nvdf_sample_trilinear(cloud_aux_previous, sp, prev_meta.xy).r;
                } else {
                    s_dp = nubis_dimensional_profile(sp, sp.y, cloud_coverage(sp.xz), stratus, h_scale).x;
                }
                let core_dens = max(0.0, s_dp - 0.05);
                let thick_boost = 1.0 + 1.15 * smoothstep(0.22, 0.72, s_dp);
                sun_depth += core_dens * segs[s] * 0.42 * thick_boost;
            }
        }
    }

    // Smooth temporal stability on voxel solar optical depth when matching same world voxel.
    if prev_valid && weather_steady {
        let pv = v + vec3<i32>(shift_xz.x, 0, shift_xz.y);
        if all(pv >= vec3(0)) && all(pv < vec3(64, 16, 64)) {
            let prev_v = textureLoad(cloud_aux_previous, nvdf_voxel_to_pixel(pv), 0);
            sun_depth = mix(sun_depth, prev_v.a, 0.50);
        }
    }

    textureStore(cloud_aux, pixel, vec4(dp_cons, center_prof.y, sdf, sun_depth));
}

fn compute(input: ComputeInput) {
    if !input.in_bounds || frame.dimension != 0 {
        return;
    }
    let cell = vec2<f32>(input.cell.xy);
    let pixel = vec2<i32>(input.cell.xy);
    let weather_key = frame.rain + 1.5 * frame.sun_direction.y;

    // Region C: 64x16x64 3D NVDF + SDF + Voxel Sun Grid ([0..1024, 768..832)).
    if pixel.y >= NVDF_ORIGIN.y {
        if pixel.x >= NVDF_ORIGIN.x && pixel.x < NVDF_ORIGIN.x + 1024 && pixel.y < NVDF_ORIGIN.y + 64 {
            nvdf_update_voxel(pixel, weather_key);
        }
        return;
    }

    // Right strip (x >= 768, y < 768): shadow map + NVDF metadata + refresh TTL.
    if pixel.x >= i32(ENV_WIDTH) {
        // Region A: Camera-centered cloud shadow & coverage map ([768..962, 0..96)).
        let local_x = pixel.x - i32(CLOUD_MAP_ORIGIN.x);
        let local_y = pixel.y - i32(CLOUD_MAP_ORIGIN.y);
        if local_x >= 0 && local_x < i32(CLOUD_MAP_WIDTH) && local_y >= 0 && local_y < i32(CLOUD_MAP_HEIGHT) {
            let old_sh = textureLoad(cloud_aux_previous, pixel, 0);
            let sh_phase = frame.frame_index & 3u;
            let sh_lane = (input.cell.x & 1u) + 2u * (input.cell.y & 1u);
            if cloud_history && sh_lane != sh_phase && old_sh.z > 0.5
                && abs(old_sh.w - weather_key) < 0.005
                && frame.delta_time > 0.0 && frame.delta_time < 0.15
                && dot(frame.camera_motion, frame.camera_motion) < 0.000225 {
                textureStore(cloud_aux, pixel, old_sh);
                return;
            }
            let shadow_uv = vec2(f32(local_x) / (CLOUD_MAP_WIDTH - 1.0), f32(local_y) / (CLOUD_MAP_HEIGHT - 1.0));
            let xz = (shadow_uv - 0.5) * CLOUD_SHADOW_EXTENT;
            let shadow = projected_cloud_shadow(vec3(xz.x, 0.0, xz.y));
            let coverage = select(0.0, cloud_coverage(frame.camera_position.xz + xz), cloud_cover > 0.0);
            textureStore(cloud_aux, pixel, vec4(shadow, coverage, 1.0, weather_key));
            return;
        }
        // Region B: NVDF anchor metadata texel (768, 200) + panorama refresh TTL (769, 200).
        if pixel.x == NVDF_META_PIXEL.x && pixel.y == NVDF_META_PIXEL.y {
            let anchor = nvdf_snap_anchor(frame.camera_position.xz);
            textureStore(cloud_aux, pixel, vec4(anchor.x, anchor.y, weather_key, 1.0));
            return;
        }
        if pixel.x == NVDF_REFRESH_PIXEL.x && pixel.y == NVDF_REFRESH_PIXEL.y {
            let prev_meta = nvdf_read_meta(cloud_aux_previous);
            let prev_ttl = textureLoad(cloud_aux_previous, NVDF_REFRESH_PIXEL, 0).x;
            let jump = length(frame.camera_motion) >= 4.0
                || (prev_meta.w > 0.5 && abs(prev_meta.z - weather_key) >= 0.015)
                || frame.delta_time >= 0.15;
            let next_ttl = select(max(prev_ttl - 1.0, 0.0), 16.0, jump);
            textureStore(cloud_aux, pixel, vec4(next_ttl, 0.0, 0.0, 1.0));
            return;
        }
        return;
    }

    // Region D: Pole-free, crease-free 768x768 Lambert Azimuthal Disk cloud cache ([0..768, 0..768)).
    // Early-out unused square corners outside the inscribed disk (saves ~18% GPU time).
    let disk_d = ((cell + 0.5 - vec2(0.5)) / vec2(ENV_WIDTH - 1.0, ENV_HEIGHT - 1.0) - vec2(0.5)) * (1.0 / CLOUD_DISK_RADIUS);
    let ray = cloud_map_direction(cell + 0.5);
    if dot(disk_d, disk_d) > 1.05 {
        textureStore(cloud_environment_map, pixel, vec4(atmosphere(ray), 1.0));
        textureStore(cloud_aux, pixel, vec4(0.0, 1.0, 0.0, weather_key));
        return;
    }
    let old = textureLoad(cloud_aux_previous, pixel, 0);
    let kind = u32(old.z);
    let active_clouds = volumetric_clouds && cloud_cover > 0.0;
    let cold_start = !cloud_history || old.y < 0.5 || !active_clouds;
    let motion_len = length(frame.camera_motion);
    let teleport = motion_len >= 4.0;
    let refresh_ttl = textureLoad(cloud_aux_previous, NVDF_REFRESH_PIXEL, 0).x;
    let refreshing = refresh_ttl > 0.5 || abs(old.w - weather_key) >= 0.015;
    let fast_change = motion_len >= 1.2 || refreshing || frame.delta_time >= 0.15;
    let reuse = !cold_start;

    let map_max = vec2<i32>(i32(ENV_WIDTH) - 1, i32(ENV_HEIGHT) - 1);
    let cam_forward = normalize((frame.view_inverse * vec4(0.0, 0.0, -1.0, 0.0)).xyz);
    let in_view = dot(ray, cam_forward) > 0.15;
    let p_c = textureLoad(cloud_environment_map_previous, pixel, 0);
    let wrapped_x = pixel.x;

    var p_l = p_c;
    var p_r = p_c;
    var p_d = p_c;
    var p_u = p_c;
    var near_cloud_edge = false;
    if in_view {
        p_l = textureLoad(cloud_environment_map_previous, clamp(pixel + vec2(-1, 0), vec2(0), map_max), 0);
        p_r = textureLoad(cloud_environment_map_previous, clamp(pixel + vec2(1, 0), vec2(0), map_max), 0);
        p_d = textureLoad(cloud_environment_map_previous, clamp(pixel + vec2(0, -1), vec2(0), map_max), 0);
        p_u = textureLoad(cloud_environment_map_previous, clamp(pixel + vec2(0, 1), vec2(0), map_max), 0);
        let edge_grad = max(abs(p_r.a - p_l.a), abs(p_u.a - p_d.a));
        let nbr_min_a = min(p_c.a, min(min(p_l.a, p_r.a), min(p_d.a, p_u.a)));
        near_cloud_edge = edge_grad > 0.12 || (old.x == 0.0 && nbr_min_a < 0.97);
    }

    var history = vec4(0.0);
    var history_distance = 0.0;
    var weight = 0.0;
    var parallax_texels = 0.0;
    if reuse && teleport {
        history = p_c;
        history_distance = old.x;
        weight = 1.0;
    } else if reuse && old.x == 0.0 {
        history = p_c;
        weight = 1.0;
    } else if reuse && !in_view {
        let wind = cloud_drift(frame.time, kind) - cloud_drift(frame.time - frame.delta_time, kind);
        let previous_point = ray * old.x + frame.camera_motion + vec3(wind.x, 0.0, wind.y);
        let previous_ray = previous_point * inverseSqrt(max(dot(previous_point, previous_point), 1e-10));
        let raw_texel = cloud_map_texel(previous_ray);
        let ang_shift = acos(clamp(dot(ray, previous_ray), -1.0, 1.0));
        parallax_texels = ang_shift * (0.5 * ENV_WIDTH / CLOUD_ELEVATION_SPAN);
        if parallax_texels < 0.12 {
            history = p_c;
        } else {
            let prev_uv = raw_texel / vec2(ENV_WIDTH, ENV_HEIGHT);
            history = textureSampleLevel(cloud_environment_map_previous, scattering_atlas_sampler, prev_uv, 0.0);
        }
        history_distance = old.x;
        weight = 1.0;
    } else if reuse {
        let wind = cloud_drift(frame.time, kind) - cloud_drift(frame.time - frame.delta_time, kind);
        let previous_point = ray * old.x + frame.camera_motion + vec3(wind.x, 0.0, wind.y);
        let previous_ray = previous_point * inverseSqrt(max(dot(previous_point, previous_point), 1e-10));
        let raw_texel = cloud_map_texel(previous_ray);
        let ang_shift = acos(clamp(dot(ray, previous_ray), -1.0, 1.0));
        parallax_texels = ang_shift * (0.5 * ENV_WIDTH / CLOUD_ELEVATION_SPAN);
        if parallax_texels < 0.22 {
            // Stationary / slow-drift clouds hold exact texel history with zero recursive blur.
            history = p_c;
            history_distance = old.x;
            weight = 1.0;
        } else {
            let texel = raw_texel - 0.5;
            let corner = vec2<i32>(floor(texel));
            let blend = fract(texel);
            var c_min = vec4(1e9);
            var c_max = vec4(-1e9);
            for (var tap = 0u; tap < 4u; tap += 1u) {
                let offset = vec2<i32>(i32(tap & 1u), i32(tap >> 1u));
                let source = clamp(corner + offset, vec2(0), map_max);
                let data = textureLoad(cloud_aux_previous, source, 0);
                if data.y < 0.5 || data.z != old.z || (data.x > 0.0) != (old.x > 0.0)
                    || abs(data.x - old.x) > max(3.0, old.x * 0.14) {
                    continue;
                }
                let axis = select(1.0 - blend, blend, offset == vec2(1));
                let w = axis.x * axis.y;
                let sample_c = textureLoad(cloud_environment_map_previous, source, 0);
                history += w * sample_c;
                history_distance += w * data.x;
                c_min = min(c_min, sample_c);
                c_max = max(c_max, sample_c);
                weight += w;
            }
            if weight >= 0.72 {
                // Replace blurry bilinear advection with clamped 5-tap Catmull-Rom.
                history = cloud_catmull_rom_history(raw_texel, c_min, c_max) * weight;
            }
        }
    }
    let valid = weight >= 0.72;
    history /= max(weight, 1e-5);
    history_distance /= max(weight, 1e-5);

    // Lightweight background panoramic refresh schedule:
    // Visible open-sky clouds now render at 1:1 native 1080p in cloud_screen_compute.wgsl,
    // so this 768x768 dome only serves reflections, skylight SH, cloud_aux distance hints,
    // and 1-frame spin disocclusion fallback (~0.25 ms GPU).
    var update = false;
    if !cloud_history || !active_clouds {
        update = true;
    } else if old.y < 0.5 {
        let phase2 = frame.frame_index & 3u;
        let lane2 = (u32(wrapped_x) & 1u) + 2u * (input.cell.y & 1u);
        if lane2 == phase2 {
            update = true;
        } else {
            let sky_fill = cloud_environment(ray, atmosphere(ray)) + aurora_volume(ray);
            textureStore(cloud_environment_map, pixel, vec4(sky_fill, 1.0));
            textureStore(cloud_aux, pixel, vec4(0.0, 0.25, 0.0, weather_key));
            return;
        }
    } else if parallax_texels > 0.75 || (in_view && refreshing) {
        let phase4 = frame.frame_index & 15u;
        update = cloud_bayer4_index(vec2<u32>(u32(wrapped_x), input.cell.y)) == phase4;
    } else if in_view && (parallax_texels > 0.25 || motion_len > 0.15) {
        let phase6 = frame.frame_index % 36u;
        let lane6 = (u32(wrapped_x) % 6u) + 6u * (input.cell.y % 6u);
        update = lane6 == phase6;
    } else if refreshing {
        let phase32 = frame.frame_index & 31u;
        let lane32 = (u32(wrapped_x) & 3u) + 4u * (input.cell.y & 7u);
        update = lane32 == phase32;
    } else {
        let phase = frame.frame_index & 63u;
        let lane = (u32(wrapped_x) & 7u) + 8u * (input.cell.y & 7u);
        let tile_phase = u32(cloud_hash(vec2(wrapped_x, pixel.y) / 8) * 64.0);
        update = ((lane + tile_phase) & 63u) == phase;
    }

    if valid && !update {
        textureStore(cloud_environment_map, pixel, history);
        textureStore(cloud_aux, pixel, vec4(history_distance, min(old.y + 1.0, 64.0), old.z, old.w));
        return;
    }

    if !in_view {
        p_l = textureLoad(cloud_environment_map_previous, clamp(pixel + vec2(-1, 0), vec2(0), map_max), 0);
        p_r = textureLoad(cloud_environment_map_previous, clamp(pixel + vec2(1, 0), vec2(0), map_max), 0);
        p_d = textureLoad(cloud_environment_map_previous, clamp(pixel + vec2(0, -1), vec2(0), map_max), 0);
        p_u = textureLoad(cloud_environment_map_previous, clamp(pixel + vec2(0, 1), vec2(0), map_max), 0);
    }

    let sky = cloud_environment(ray, atmosphere(ray)) + aurora_volume(ray);
    // True 512x512 isotropic blue noise (photon_noise.b) + R1 sequence (zero diagonal IGN bands).
    let bn = textureLoad(photon_noise, pixel & vec2(511), 0).b;
    let cycle = select(frame.frame_index >> 5u, select(frame.frame_index >> 4u, frame.frame_index >> 2u, near_cloud_edge || fast_change || parallax_texels > 0.20), in_view);
    let dither = fract(bn + f32(cycle) * 0.61803399);
    let view_steps = select(16u, 20u, ray.y < 0.28);
    let steps = select(10u, view_steps, in_view);
    let clouds = cloud_layers(cloud_aux_previous, ray, 12000.0, steps, dither);
    let radiance = sky * clouds.transmittance + clouds.color + cloud_sunbeams(cloud_aux_previous, ray, dither);
    let fresh = vec4(radiance, clouds.transmittance);
    let match_depth = abs(clouds.distance - history_distance) < max(48.0, 0.22 * clouds.distance);
    let same_hit = (clouds.distance > 0.0) == (old.x > 0.0);

    // Salvi variance-guided neighborhood clipping (mean +/- gamma*sigma) across the 5-tap
    // cross rejects stale ghosting history tightly while keeping crisp cloud silhouettes.
    let m1 = (fresh + p_l + p_r + p_d + p_u) * 0.2;
    let m2 = (fresh * fresh + p_l * p_l + p_r * p_r + p_d * p_d + p_u * p_u) * 0.2;
    let sigma = sqrt(max(m2 - m1 * m1, vec4(0.0)));
    let speed = motion_len / max(frame.delta_time, 0.004);
    let motion_clamp = clamp(speed * 0.16, 0.0, 1.0);
    let flow_clamp = clamp(max(motion_clamp, parallax_texels * 1.15), 0.0, 1.0);
    let gamma = mix(1.15, 0.65, flow_clamp);
    let var_min = max(m1 - sigma * gamma, vec4(0.0));
    let var_max = m1 + sigma * gamma;
    let limited_history = vec4(clamp(history.rgb, var_min.rgb, var_max.rgb), clamp(history.a, var_min.a, min(var_max.a, 1.0)));

    // Fresh-dominant accumulation: cap stationary blend at 0.55 (not 0.90) to denoise
    // dither without smearing, and drop to 0.0 (100% fresh) under camera/parallax motion.
    let static_w = clamp(1.0 - 1.0 / max(old.y * 0.15 + 1.25, 1.25), 0.20, 0.55);
    var blend_w = select(0.0, static_w, valid && match_depth && clouds.layer == kind && same_hit);
    let flow_damp = 1.0 - smoothstep(0.08, 0.35, parallax_texels);
    let move_damp = 1.0 - smoothstep(0.03, 0.18, motion_len);
    let alpha_jump = 1.0 - smoothstep(0.05, 0.16, abs(fresh.a - history.a));
    blend_w *= flow_damp * move_damp * alpha_jump;
    if teleport || refreshing || !cloud_history {
        blend_w = 0.0;
    }
    let resolved = mix(fresh, limited_history, blend_w);
    let next_age = select(1.0, min(old.y + 1.0, 64.0), blend_w > 0.08);
    textureStore(cloud_environment_map, pixel, resolved);
    textureStore(cloud_aux, pixel, vec4(clouds.distance, next_age, f32(clouds.layer), weather_key));
}
