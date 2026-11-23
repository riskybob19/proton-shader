override cloud_history: bool = true;

fn cloud_coverage(world: vec2<f32>) -> f32 {
    return cumulus_coverage(world);
}

fn compute(input: ComputeInput) {
    if !input.in_bounds {
        return;
    }
    let cell = vec2<f32>(input.cell.xy);
    let pixel = vec2<i32>(input.cell.xy);
    if frame.dimension != 0 {
        return;
    }
    let weather_key = frame.rain + 1.5 * frame.sun_direction.y;
    if cell.y >= ENV_HEIGHT {
        if cell.x >= CLOUD_MAP_WIDTH {
            return;
        }
        // Amortize the 194x96 camera-centered cloud shadow map across 4 frames (2x2 staggered)
        // only when the camera is stationary and sun/weather are unchanged (zero motion checkerboard).
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
        let shadow_uv = vec2(cell.x / (CLOUD_MAP_WIDTH - 1.0), (cell.y - ENV_HEIGHT) / (CLOUD_MAP_HEIGHT - 1.0));
        let xz = (shadow_uv - 0.5) * CLOUD_SHADOW_EXTENT;
        let shadow = projected_cloud_shadow(vec3(xz.x, 0.0, xz.y));
        let coverage = select(0.0, cloud_coverage(frame.camera_position.xz + xz), cloud_cover > 0.0);
        textureStore(cloud_aux, pixel, vec4(shadow, coverage, 1.0, weather_key));
        return;
    }

    if cell.x >= ENV_WIDTH {
        return;
    }
    let ray = cloud_map_direction(cell + 0.5);
    let old = textureLoad(cloud_aux_previous, pixel, 0);
    let kind = u32(old.z);
    let reuse = cloud_history && volumetric_clouds && cloud_cover > 0.0 && old.y >= 1.0
        && frame.delta_time > 0.0 && frame.delta_time < 0.15
        && length(frame.camera_motion) < 1.8;

    let map_max = vec2<i32>(i32(ENV_WIDTH) - 1, i32(ENV_HEIGHT) - 1);
    let cam_forward = normalize((frame.view_inverse * vec4(0.0, 0.0, -1.0, 0.0)).xyz);
    let in_view = dot(ray, cam_forward) > 0.15;
    let p_c = textureLoad(cloud_environment_map_previous, pixel, 0);
    let wrapped_x = (pixel.x + i32(ENV_WIDTH) - 6) % i32(ENV_WIDTH - 4.0);

    // Fast early-out for rear-hemisphere texels (~60% of the environment map):
    // skip the 4-neighbor edge loads when behind the camera and not on their 8x8 update slot.
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
        near_cloud_edge = edge_grad > 0.16 || (old.x == 0.0 && nbr_min_a < 0.85);
    }

    var history = vec4(0.0);
    var history_distance = 0.0;
    var weight = 0.0;
    var prev_uv = vec2(0.0);
    if reuse && old.x == 0.0 {
        // Empty sky has no parallax; mark valid unless adjacent to an advancing cloud edge.
        if abs(old.w - weather_key) < 0.018 {
            history = p_c;
            weight = 1.0;
        }
    } else if reuse && !in_view {
        // Rear hemisphere (~60% of map): 1 hardware bilinear fetch instead of 8 manual loads.
        if abs(old.w - weather_key) < 0.018 {
            let wind = cloud_drift(frame.time, kind) - cloud_drift(frame.time - frame.delta_time, kind);
            let previous_point = ray * old.x + frame.camera_motion + vec3(wind.x, 0.0, wind.y);
            let previous_ray = previous_point * inverseSqrt(max(dot(previous_point, previous_point), 1e-10));
            prev_uv = cloud_map_texel(previous_ray) / vec2(ENV_WIDTH, ENV_HEIGHT);
            history = textureSampleLevel(cloud_environment_map_previous, scattering_atlas_sampler, prev_uv, 0.0);
            history_distance = old.x;
            weight = 1.0;
        }
    } else if reuse {
        let wind = cloud_drift(frame.time, kind) - cloud_drift(frame.time - frame.delta_time, kind);
        let previous_point = ray * old.x + frame.camera_motion + vec3(wind.x, 0.0, wind.y);
        let previous_ray = previous_point * inverseSqrt(max(dot(previous_point, previous_point), 1e-10));
        let raw_texel = cloud_map_texel(previous_ray);
        prev_uv = raw_texel / vec2(ENV_WIDTH, ENV_HEIGHT);
        let texel = raw_texel - 0.5;
        let corner = vec2<i32>(floor(texel));
        let blend = fract(texel);
        for (var tap = 0u; tap < 4u; tap += 1u) {
            let offset = vec2<i32>(i32(tap & 1u), i32(tap >> 1u));
            let source = clamp(corner + offset, vec2(0), map_max);
            let data = textureLoad(cloud_aux_previous, source, 0);
            if data.y < 1.0 || data.z != old.z || (data.x > 0.0) != (old.x > 0.0)
                || abs(data.x - old.x) > max(4.0, old.x * 0.22) || abs(data.w - weather_key) >= 0.018 {
                continue;
            }
            let axis = select(1.0 - blend, blend, offset == vec2(1));
            let w = axis.x * axis.y;
            history += w * textureLoad(cloud_environment_map_previous, source, 0);
            history_distance += w * data.x;
            weight += w;
        }
    }
    let valid = weight >= 0.5;
    history /= max(weight, 1e-5);
    history_distance /= max(weight, 1e-5);

    // Wide forward hemisphere (dot > 0.15) prevents pop-in during fast camera turns.
    // Edge-triggered 6x6 (36-frame = 0.30s @ 120 FPS) refresh on cloud silhouettes keeps edges
    // crisp while clear sky and solid interiors amortize across 8x8 (64-frame = 0.53s @ 120 FPS).
    var update = false;
    if in_view {
        if near_cloud_edge {
            let phase6 = frame.frame_index % 36u;
            let lane6 = (u32(wrapped_x) % 6u) + 6u * (input.cell.y % 6u);
            update = lane6 == phase6;
        } else {
            let phase = frame.frame_index & 63u;
            let lane = (u32(wrapped_x) & 7u) + 8u * (input.cell.y & 7u);
            let tile_phase = u32(cloud_hash(vec2(wrapped_x, pixel.y) / 8) * 64.0);
            update = ((lane + tile_phase) & 63u) == phase;
        }
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
    let nbr_min = min(p_c, min(min(p_l, p_r), min(p_d, p_u)));
    let nbr_max = max(p_c, max(max(p_l, p_r), max(p_d, p_u)));

    let sky = cloud_environment(ray, atmosphere(ray)) + aurora_volume(ray);
    // Interleaved Gradient Noise + R1 temporal sequence breaks up horizontal
    // altitude slicing cleanly without structured texture noise.
    let ign = fract(52.9829189 * fract(dot(vec2<f32>(pixel), vec2(0.06711056, 0.00583715))));
    let cycle = select(frame.frame_index >> 5u, select(frame.frame_index >> 4u, frame.frame_index >> 3u, near_cloud_edge), in_view);
    let dither = fract(ign + f32(cycle) * 0.61803399);
    let view_steps = select(12u, 16u, ray.y < 0.28);
    let steps = select(8u, view_steps, in_view);
    let clouds = cloud_layers(ray, 12000.0, steps, dither);
    let radiance = sky * clouds.transmittance + clouds.color + cloud_sunbeams(ray, dither);
    let fresh = vec4(radiance, clouds.transmittance);
    let match_depth = abs(clouds.distance - history_distance) < max(64.0, 0.28 * clouds.distance);
    let same_hit = (clouds.distance > 0.0) == (old.x > 0.0);
    let on_edge = near_cloud_edge || (fresh.a > 0.04 && fresh.a < 0.995) || (history.a > 0.04 && history.a < 0.995);

    // Spatial 5-tap cross neighborhood AABB clamping (Frostbite / UE5 / Nubis):
    // clips stale moving-cloud ghosts in 1 frame while preserving smooth static convergence.
    let speed = length(frame.camera_motion) / max(frame.delta_time, 0.004);
    let motion_clamp = clamp(speed * 0.12, 0.0, 1.0);
    let aabb_min = min(fresh, nbr_min);
    let aabb_max = max(fresh, nbr_max);
    let pad = mix(0.14, 0.03, motion_clamp);
    let clamped_rgb = clamp(history.rgb,
        max(aabb_min.rgb - vec3(pad) * max(radiance, vec3(0.08)), vec3(0.0)),
        aabb_max.rgb + vec3(pad) * max(radiance, vec3(0.08)));
    let clamped_a = clamp(history.a,
        max(aabb_min.a - mix(0.18, 0.05, motion_clamp), 0.0),
        min(aabb_max.a + mix(0.18, 0.05, motion_clamp), 1.0));
    let limited_history = vec4(clamped_rgb, clamped_a);

    // Age-based accumulation with faster convergence on moving cloud silhouettes.
    let base_age = clamp(1.0 - 1.0 / max(old.y * 0.25 + 1.5, 1.5), 0.60, 0.90);
    let age_weight = mix(base_age, min(base_age, 0.72), motion_clamp * select(0.35, 1.0, on_edge));
    let blend_w = select(
        select(0.0, age_weight * 0.78, valid && on_edge),
        age_weight,
        valid && match_depth && clouds.layer == kind && same_hit,
    );
    let resolved = mix(fresh, limited_history, blend_w);
    let next_age = select(1.0, min(old.y + 1.0, 64.0), blend_w > 0.1);
    textureStore(cloud_environment_map, pixel, resolved);
    textureStore(cloud_aux, pixel, vec4(clouds.distance, next_age, f32(clouds.layer), weather_key));
}
