override cloud_history: bool = true;

fn cloud_coverage(world: vec2<f32>) -> f32 {
    return cumulus_coverage(world);
}

// 4x4 Bayer order (0..15): every 2x2 quad updates every 4 frames and the entire
// native-resolution screen sky refreshes in 16 frames (Photon / Nubis3 / RDR2).
fn screen_cloud_bayer4(p: vec2<u32>) -> u32 {
    let x = p.x & 3u;
    let y = p.y & 3u;
    let b0 = ((x & 1u) ^ (y & 1u)) | ((y & 1u) << 1u);
    let b1 = (((x >> 1u) & 1u) ^ ((y >> 1u) & 1u)) | (((y >> 1u) & 1u) << 1u);
    return (b0 << 2u) | b1;
}

fn pack_screen_cloud(c: vec4<f32>) -> vec4<u32> {
    return pack_u32_vec4(vec4(pack_hdr8(c.rgb), 0.02 + 0.98 * clamp(c.a, 0.0, 1.0)));
}

fn unpack_screen_cloud_raw(raw: vec4<f32>) -> vec4<f32> {
    return vec4(unpack_hdr8(raw.rgb), clamp((raw.a - 0.02) * (1.0 / 0.98), 0.0, 1.0));
}

fn load_prev_screen_cloud(px: vec2<i32>) -> vec4<f32> {
    return unpack_u32_vec4(textureLoad(cloud_screen_history_previous, px, 0));
}

// 12-tap cross Catmull-Rom filter over r32uint screen-space cloud history (packed perceptual space).
// Uses convex bilinear interpolation across sharp transmittance silhouettes so negative Catmull-Rom
// lobes can never decouple rgb -> 0 from transmittance -> 0 (eliminates black edge dots).
fn screen_cloud_catmull_rom(uv: vec2<f32>, size: vec2<i32>, c_min: vec4<f32>, c_max: vec4<f32>) -> vec4<f32> {
    let res = vec2<f32>(size);
    let pos = uv * res - 0.5;
    let base = vec2<i32>(floor(pos));
    let f = clamp(fract(pos), vec2(0.0), vec2(1.0));
    let max_px = size - 1;
    let b00 = load_prev_screen_cloud(clamp(base, vec2(0), max_px));
    let b10 = load_prev_screen_cloud(clamp(base + vec2(1, 0), vec2(0), max_px));
    let b01 = load_prev_screen_cloud(clamp(base + vec2(0, 1), vec2(0), max_px));
    let b11 = load_prev_screen_cloud(clamp(base + vec2(1, 1), vec2(0), max_px));
    let bi = mix(mix(b00, b10, f.x), mix(b01, b11, f.x), f.y);
    if (c_max.a - c_min.a) > 0.08 {
        return unpack_screen_cloud_raw(bi);
    }
    let f2 = f * f;
    let f3 = f2 * f;
    let wx = vec4(
        -0.5 * f3.x + f2.x - 0.5 * f.x,
         1.5 * f3.x - 2.5 * f2.x + 1.0,
        -1.5 * f3.x + 2.0 * f2.x + 0.5 * f.x,
         0.5 * f3.x - 0.5 * f2.x,
    );
    let wy = vec4(
        -0.5 * f3.y + f2.y - 0.5 * f.y,
         1.5 * f3.y - 2.5 * f2.y + 1.0,
        -1.5 * f3.y + 2.0 * f2.y + 0.5 * f.y,
         0.5 * f3.y - 0.5 * f2.y,
    );
    var sum = vec4(0.0);
    var norm = 0.0;
    for (var iy = -1; iy <= 2; iy += 1) {
        let wy_i = wy[iy + 1];
        for (var ix = -1; ix <= 2; ix += 1) {
            if (ix == -1 || ix == 2) && (iy == -1 || iy == 2) {
                continue;
            }
            let s = load_prev_screen_cloud(clamp(base + vec2(ix, iy), vec2(0), max_px));
            if s.a < 0.01 {
                return unpack_screen_cloud_raw(bi);
            }
            let w = wx[ix + 1] * wy_i;
            sum += s * w;
            norm += w;
        }
    }
    let sharp = clamp(max(sum / max(norm, 1e-5), vec4(0.0)), c_min, c_max);
    return unpack_screen_cloud_raw(sharp);
}

// Extract cloud-only (rgb = cloud radiance, a = transmittance) from the 768x768 panoramic dome
// for zero-artifact fallback on newly disoccluded screen edges during rapid camera spins.
fn pano_cloud_fallback(ray: vec3<f32>) -> vec4<f32> {
    let pano = sample_environment(ray);
    let base_atmo = atmosphere(ray);
    let cloud_only = max(pano.rgb - base_atmo * pano.a, vec3(0.0));
    return vec4(cloud_only, clamp(pano.a, 0.0, 1.0));
}

fn compute(input: ComputeInput) {
    let size = vec2<i32>(textureDimensions(cloud_screen_history));
    let pixel = vec2<i32>(input.cell.xy);
    if !input.in_bounds || any(pixel >= size) {
        return;
    }

    // Only Overworld air/water sky pixels need full-res screen clouds; store terrain sentinel (0.0 in .a).
    let active_clouds = frame.dimension == 0 && frame.eye_in < 2u && volumetric_clouds && cloud_cover > 0.0;
    if !active_clouds {
        textureStore(cloud_screen_history, pixel, pack_screen_cloud(vec4(0.0, 0.0, 0.0, 1.0)));
        return;
    }

    let depth = textureLoad(scene_depth, pixel, 0);
    if depth > 0.0 {
        // Terrain-occluded pixel: early-out immediately (cuts raymarch cost ~40-60%)
        // and write 0u alpha sentinel so reprojection detects terrain disocclusions.
        textureStore(cloud_screen_history, pixel, vec4<u32>(0u));
        return;
    }

    let res = vec2<f32>(size);
    let uv = (vec2<f32>(pixel) + 0.5) / res;
    // Camera-rotation-locked, translation-free world ray (zero view-bob sway, exact terrain lock).
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray = sky_world_ray(ndc);

    let above_clouds = frame.camera_position.y >= CLOUD_BASE - 8.0;
    if !above_clouds && ray.y < -0.075 {
        textureStore(cloud_screen_history, pixel, pack_screen_cloud(vec4(0.0, 0.0, 0.0, 1.0)));
        return;
    }

    // Smooth per-pixel analytic cloud-shell intersection distance (works looking both up
    // and down to ray.y = -1.0), refined with cached panoramic hit depth when inside the dome.
    // Continuous across cloud silhouettes so fast falling/flight never creates depth jumps.
    let shell = cloud_shell(ray, 12000.0, vec2(CLOUD_BASE, CLOUD_TOP));
    let has_shell = shell.y > shell.x;
    let d_shell = select(0.0, shell.x + min((shell.y - shell.x) * 0.32, 160.0), has_shell);
    var d_hit = d_shell;
    var kind = 0u;
    if ray.y >= -0.072 {
        let map_max = vec2<i32>(i32(ENV_WIDTH) - 1, i32(ENV_HEIGHT) - 1);
        let pano_cell = clamp(vec2<i32>(cloud_map_texel(ray)), vec2(0), map_max);
        let pano_aux = textureLoad(cloud_aux, pano_cell, 0);
        if pano_aux.x > 0.0 {
            kind = u32(pano_aux.z + 0.5);
            if has_shell && kind == 0u {
                d_hit = clamp(pano_aux.x, shell.x, min(shell.y, shell.x + 800.0));
            } else {
                d_hit = pano_aux.x;
            }
        }
    }

    let motion_len = length(frame.camera_motion);
    let teleport = motion_len >= 4.0;
    let refresh_ttl = textureLoad(cloud_aux, NVDF_REFRESH_PIXEL, 0).x;
    let refreshing = refresh_ttl > 0.5;

    // Direction-only reprojection (w = 0.0): preserves previous-frame camera rotation R_bob
    // while cancelling 100% of bobView eye translation (zero walking wobble).
    var prev_world = ray;
    if d_hit > 0.0 {
        let wind = cloud_drift(frame.time, kind) - cloud_drift(frame.time - frame.delta_time, kind);
        prev_world = ray * d_hit + frame.camera_motion + vec3(wind.x, 0.0, wind.y);
    }

    var disoccluded = !cloud_history || teleport || refreshing;
    var prev_uv = uv;
    var vel_px = vec2(0.0);
    var motion_px = 0.0;
    if !disoccluded {
        let prev_clip = frame.previous_view_projection * vec4(prev_world, 0.0);
        if prev_clip.w <= 1e-4 {
            disoccluded = true;
        } else {
            let prev_ndc = prev_clip.xy / prev_clip.w;
            prev_uv = 0.5 + prev_ndc * vec2(0.5, -0.5);
            if any(prev_uv <= vec2(0.5) / res) || any(prev_uv >= vec2(1.0) - vec2(0.5) / res) {
                disoccluded = true;
            } else {
                vel_px = (uv - prev_uv) * res;
                motion_px = length(vel_px);
            }
        }
    }

    var history = vec4(0.0, 0.0, 0.0, 1.0);
    var p_min = vec4(1e9);
    var p_max = vec4(-1e9);
    var is_edge = false;
    if !disoccluded {
        let p_pos = prev_uv * res - 0.5;
        let corner = clamp(vec2<i32>(floor(p_pos)), vec2(0), size - 2);
        let c00 = load_prev_screen_cloud(corner);
        let c10 = load_prev_screen_cloud(corner + vec2(1, 0));
        let c01 = load_prev_screen_cloud(corner + vec2(0, 1));
        let c11 = load_prev_screen_cloud(corner + vec2(1, 1));
        // Any terrain sentinel (< 0.01) in the 2x2 footprint invalidates screen history cleanly.
        if c00.a < 0.01 || c10.a < 0.01 || c01.a < 0.01 || c11.a < 0.01 {
            disoccluded = true;
        } else {
            let raw_min = min(min(c00, c10), min(c01, c11));
            let raw_max = max(max(c00, c10), max(c01, c11));
            p_min = unpack_screen_cloud_raw(raw_min);
            p_max = unpack_screen_cloud_raw(raw_max);

            // Track moving cloud silhouettes across a motion-swept +/-5px footprint so
            // advancing edges during fast flight/falling never lag behind into Bayer comb teeth.
            var band_min_a = raw_min.a;
            var band_max_a = raw_max.a;
            if motion_len > 0.05 || motion_px > 0.8 {
                let p_center = clamp(vec2<i32>(prev_uv * res), vec2(0), size - 1);
                let max_px = size - 1;
                let e0 = load_prev_screen_cloud(clamp(p_center + vec2(0, -5), vec2(0), max_px)).a;
                let e1 = load_prev_screen_cloud(clamp(p_center + vec2(0, 5), vec2(0), max_px)).a;
                let e2 = load_prev_screen_cloud(clamp(p_center + vec2(-5, 0), vec2(0), max_px)).a;
                let e3 = load_prev_screen_cloud(clamp(p_center + vec2(5, 0), vec2(0), max_px)).a;
                let dv = clamp(vec2<i32>(round(vel_px * 1.6)), vec2(-14), vec2(14));
                let e4 = load_prev_screen_cloud(clamp(p_center - dv, vec2(0), max_px)).a;
                if e0 >= 0.01 { band_min_a = min(band_min_a, e0); band_max_a = max(band_max_a, e0); }
                if e1 >= 0.01 { band_min_a = min(band_min_a, e1); band_max_a = max(band_max_a, e1); }
                if e2 >= 0.01 { band_min_a = min(band_min_a, e2); band_max_a = max(band_max_a, e2); }
                if e3 >= 0.01 { band_min_a = min(band_min_a, e3); band_max_a = max(band_max_a, e3); }
                if e4 >= 0.01 { band_min_a = min(band_min_a, e4); band_max_a = max(band_max_a, e4); }
            }
            is_edge = (band_max_a - band_min_a) > 0.025 || (band_min_a < 0.96 && band_max_a > 0.05);

            if motion_px < 0.06 {
                history = unpack_screen_cloud_raw(load_prev_screen_cloud(clamp(vec2<i32>(prev_uv * res), vec2(0), size - 1)));
            } else {
                history = screen_cloud_catmull_rom(prev_uv, size, raw_min, raw_max);
            }
        }
    }

    // 4x4 Bayer schedule (1/16 native pixels/frame); ramps to 2x2 on disocclusions/close
    // parallax and to full 1x1 on fast-moving cloud silhouettes to eliminate comb teeth.
    let bayer = screen_cloud_bayer4(vec2<u32>(pixel));
    let phase4 = frame.frame_index & 15u;
    var update = bayer == phase4;
    let fast_close = d_hit > 0.0 && d_hit < 360.0 && motion_len > 0.20;
    let med_edge = is_edge && (motion_len > 0.06 || motion_px > 0.9);
    let fast_edge = is_edge && (motion_len > 0.16 || motion_px > 2.2);
    if !cloud_history || disoccluded || fast_close || med_edge {
        let lane2 = (u32(pixel.x) & 1u) + 2u * (u32(pixel.y) & 1u);
        update = lane2 == (frame.frame_index & 3u);
    }
    if fast_edge || (disoccluded && ray.y < -0.072) {
        update = true;
    }

    if !update {
        if disoccluded {
            // Hybrid secret weapon: seamless 768x768 panoramic cloud fallback on disoccluded
            // screen edges above the dome horizon (zero 4x4 blocks during fast camera spins!).
            textureStore(cloud_screen_history, pixel, pack_screen_cloud(pano_cloud_fallback(ray)));
        } else {
            textureStore(cloud_screen_history, pixel, pack_screen_cloud(history));
        }
        return;
    }

    // Fresh native 1.0x (1920x1080) Nubis3 + NVDF + v3 Neural MLP raymarch.
    // Use true 512x512 isotropic blue noise (photon_noise.b) + R1 golden-ratio sequence
    // instead of IGN, which forms ~55-degree diagonal iso-contour stripes on a 4x4 grid.
    let bn = textureLoad(photon_noise, pixel & vec2(511), 0).b;
    let cycle = select(
        select(frame.frame_index >> 4u, frame.frame_index >> 2u, disoccluded || fast_close || med_edge),
        frame.frame_index,
        fast_edge,
    );
    let dither = fract(bn + f32(cycle) * 0.61803399);
    let steps = select(20u, 26u, abs(ray.y) < 0.28);
    let clouds = cloud_layers(cloud_aux, ray, 12000.0, steps, dither);
    let sunbeams = cloud_sunbeams(cloud_aux, ray, dither);
    let fresh = vec4(max(clouds.color + sunbeams, vec3(0.0)), clamp(clouds.transmittance, 0.0, 1.0));

    if disoccluded || !cloud_history {
        textureStore(cloud_screen_history, pixel, pack_screen_cloud(fresh));
        return;
    }

    // Photon-style 3x3 cross/box HRAA soft min/max clamping + off-center rejection:
    // eliminates trailing ghosts and bilinear smear while converging raymarch dither cleanly.
    let n_l = load_prev_screen_cloud(clamp(pixel + vec2(-1, 0), vec2(0), size - 1));
    let n_r = load_prev_screen_cloud(clamp(pixel + vec2(1, 0), vec2(0), size - 1));
    let n_d = load_prev_screen_cloud(clamp(pixel + vec2(0, -1), vec2(0), size - 1));
    let n_u = load_prev_screen_cloud(clamp(pixel + vec2(0, 1), vec2(0), size - 1));
    let s_l = select(fresh, unpack_screen_cloud_raw(n_l), n_l.a >= 0.01);
    let s_r = select(fresh, unpack_screen_cloud_raw(n_r), n_r.a >= 0.01);
    let s_d = select(fresh, unpack_screen_cloud_raw(n_d), n_d.a >= 0.01);
    let s_u = select(fresh, unpack_screen_cloud_raw(n_u), n_u.a >= 0.01);

    let cross_min = min(fresh, min(min(s_l, s_r), min(s_d, s_u)));
    let cross_max = max(fresh, max(max(s_l, s_r), max(s_d, s_u)));
    let soft_min = 0.5 * (cross_min + min(cross_min, p_min));
    let soft_max = 0.5 * (cross_max + max(cross_max, p_max));

    let dt = max(frame.delta_time, 0.004);
    let speed_sq = dot(frame.camera_motion, frame.camera_motion) / (dt * dt);
    let dist_eff = select(max(d_hit, 25.0), 4000.0, d_hit <= 0.0);
    let vel_raw = 75.0 * speed_sq / max(dist_eff * dist_eff, 1.0);
    let vel_factor = clamp(max(vel_raw / (vel_raw + 1.0), smoothstep(1.5, 10.0, motion_px) * 0.85), 0.0, 1.0);

    // Coupled scalar ray-clipping from `history` toward physically-valid `fresh`:
    // every point on the segment fresh -> history keeps rgb radiance and transmittance
    // strictly coupled, making black opaque edge dots mathematically impossible.
    let delta_h = history - fresh;
    let hi_t = select(vec4(1.0), (soft_max - fresh) / max(delta_h, vec4(1e-5)), delta_h > vec4(1e-5));
    let lo_t = select(vec4(1.0), (soft_min - fresh) / min(delta_h, vec4(-1e-5)), delta_h < vec4(-1e-5));
    let t_clip = clamp(min(min(min(hi_t.r, hi_t.g), min(hi_t.b, hi_t.a)), min(min(lo_t.r, lo_t.g), min(lo_t.b, lo_t.a))), 0.0, 1.0);
    let clamped_hist = fresh + delta_h * t_clip;
    let limited_hist = mix(history, clamped_hist, max(vel_factor, 0.55));

    // Photon off-center rejection: reduces weight when prev_uv lands halfway between texels.
    let px_center_off = vec2(1.0) - abs(fract(prev_uv * res) * 2.0 - 1.0);
    let offcenter = sqrt(clamp(px_center_off.x * px_center_off.y, 0.0, 1.0));

    let alpha_jump = 1.0 - smoothstep(0.05, 0.20, abs(fresh.a - history.a));
    let move_damp = 1.0 - smoothstep(0.05, 0.28, motion_len);
    let px_damp = 1.0 - smoothstep(1.0, 4.5, motion_px);
    let edge_damp = select(1.0, 0.30, fast_edge);
    let blend_w = 0.78 * mix(1.0, offcenter, 0.65) * alpha_jump
        * mix(0.20, 1.0, move_damp) * mix(0.25, 1.0, px_damp) * edge_damp;
    let resolved = max(mix(fresh, limited_hist, blend_w), vec4(0.0));
    textureStore(cloud_screen_history, pixel, pack_screen_cloud(resolved));
}
