// Dedicated half-resolution SVGF Temporal GI + Hardware-RT Emissive ReSTIR DI + Variance-Guided Shadow Denoiser:
//   .rgb = Motion-reprojected, 5-tap variance-clamped 3-layer Hybrid Voxel GI (indirect_gi)
//          + Hardware-RT LPV-Guided Emissive ReSTIR DI (shadowed spherical 1/r^2 torch/lava lighting)
//   .a   = Motion-reprojected, moment/variance-clamped + depth-aware denoised distant shadow
//          (preserves razor-sharp <24m contact shadows while eliminating distant shadow dither).

// Walk 3 gradient-ascent hops inside the 26-neighbor L2 LPV to locate the dominant
// glowing emitter voxel around a starting world offset (finds even tiny 0.125m torches!).
fn lpv_ascent_peak(start_world: vec3<f32>, anchor: vec3<f32>) -> vec4<f32> {
    var cur = clamp(
        vec3<i32>(floor(start_world * (1.0 / LPV_CELL) - anchor + vec3(0.5 * LPV_XZ, 0.5 * LPV_Y, 0.5 * LPV_XZ))),
        vec3(0), vec3(63, 31, 63),
    );
    var best_rgb = max(textureLoad(world_volumes, lpv_voxel_to_pixel(cur), 0).rgb, vec3(0.0));
    var best_lum = dot(best_rgb, frame.luminance);
    for (var step = 0u; step < 3u; step += 1u) {
        var next_v = cur;
        var next_rgb = best_rgb;
        var next_lum = best_lum;
        for (var i = 0u; i < 6u; i += 1u) {
            var off = vec3<i32>(0);
            if i == 0u { off.x = -1; } else if i == 1u { off.x = 1; }
            else if i == 2u { off.y = -1; } else if i == 3u { off.y = 1; }
            else if i == 4u { off.z = -1; } else { off.z = 1; }
            let q = clamp(cur + off, vec3(0), vec3(63, 31, 63));
            let c = max(textureLoad(world_volumes, lpv_voxel_to_pixel(q), 0).rgb, vec3(0.0));
            let lum = dot(c, frame.luminance);
            if lum > next_lum * 1.03 {
                next_lum = lum;
                next_v = q;
                next_rgb = c;
            }
        }
        if all(next_v == cur) {
            break;
        }
        cur = next_v;
        best_rgb = next_rgb;
        best_lum = next_lum;
    }
    if best_lum <= 0.025 {
        return vec4(0.0);
    }
    let peak_world = (anchor + vec3<f32>(cur) - vec3(0.5 * LPV_XZ, 0.5 * LPV_Y, 0.5 * LPV_XZ) + 0.5) * LPV_CELL;
    return vec4(peak_world, best_lum);
}

// LPV-Guided Hardware-RT Emissive ReSTIR DI:
//   1. Proposes 3 spatially-jittered LPV-peak emitter candidates + 1 local RT hemisphere probe
//   2. Selects 1 winner via Resampled Importance Sampling (RIS) weighted by N.L / r^2
//   3. Casts 1 hardware VK_KHR_ray_query (trace(), RAY_SOLID) visibility ray for crisp torch shadows
fn rt_emissive_restir_di(pos: vec3<f32>, normal: vec3<f32>, cell: vec2<i32>, view_dist: f32) -> vec3<f32> {
    if blocklight_mode == 0u || frame.has_ray_tracing == 0u || view_dist >= 88.0 {
        return vec3(0.0);
    }
    let anchor = lpv_anchor(frame.camera_position);
    let world_pos = frame.camera_position + pos;
    let center_lpv = lpv_sample_trilinear(world_volumes, world_pos, anchor);
    let center_lum = dot(center_lpv, frame.luminance);

    let ign = fract(52.9829189 * fract(dot(vec2<f32>(cell), vec2(0.06711056, 0.00583715))));
    let seed = fract(ign + f32(frame.frame_index & 31u) * 0.61803399);
    let rand2 = fract(vec2(seed, seed * 1.32471796) + vec2(0.17, 0.43));
    let tangent = normalize(select(vec3(1.0, 0.0, 0.0), cross(vec3(0.0, 1.0, 0.0), normal), abs(normal.y) < 0.92));
    let bitangent = cross(normal, tangent);

    var w_sum = 0.0;
    var win_pos = vec3(0.0);
    var win_col = vec3(0.0);
    var win_phat = 0.0;
    var win_confirmed = false;
    var candidates = 0.0;

    // 3 R2-jittered gradient-ascent proposals across colored_light_volume (captures multi-torch streets).
    if center_lum > 0.004 {
        for (var k = 0u; k < 3u; k += 1u) {
            let ang = (f32(k) + rand2.x) * (TAU / 3.0);
            let rad = mix(0.0, 4.5, f32(k) * 0.45 + 0.15 * rand2.y);
            let start_w = world_pos + (tangent * cos(ang) + bitangent * sin(ang)) * rad + normal * 0.6;
            let peak = lpv_ascent_peak(start_w, anchor);
            if peak.w <= 0.025 {
                continue;
            }
            // Small sub-voxel jitter turns voxel centers into natural soft penumbras.
            let sub_jit = (tangent * (rand2.x - 0.5) + bitangent * (rand2.y - 0.5)) * 0.75;
            let emit_rel = (peak.xyz + sub_jit) - frame.camera_position;
            let to_l = emit_rel - pos;
            let d2 = dot(to_l, to_l);
            if d2 < 0.04 || d2 > 625.0 {
                continue;
            }
            let dist = sqrt(d2);
            let dir = to_l / dist;
            let ndotl = max(dot(normal, dir), 0.08);
            if ndotl <= 0.01 {
                continue;
            }
            let emit_rgb = max(lpv_sample_trilinear(world_volumes, peak.xyz, anchor), vec3(0.0));
            let emit_l = dot(emit_rgb, frame.luminance);
            let p_hat = (emit_l * ndotl) / (0.85 + 0.28 * d2);
            let xi = fract(seed + f32(k) * 0.381966 + 0.19);
            w_sum += p_hat;
            candidates += 1.0;
            if xi * max(w_sum, 1e-6) <= p_hat {
                win_pos = emit_rel;
                win_col = emit_rgb;
                win_phat = p_hat;
                win_confirmed = false;
            }
        }
    }

    // Candidate 4: 1 local hardware-RT hemisphere discovery probe for unseen nearby glowing blocks.
    let phi = TAU * rand2.x;
    let cos_t = sqrt(1.0 - 0.85 * rand2.y);
    let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
    let ray_dir = normalize(tangent * (cos(phi) * sin_t) + bitangent * (sin(phi) * sin_t) + normal * cos_t);
    let origin = pos + normal * clamp(0.05 + view_dist * 0.001, 0.05, 0.14);
    let disc = trace(origin, ray_dir, 14.0);
    if disc.distance > 0.25 && disc.emission > 0.05 {
        let d2 = max(disc.distance * disc.distance, 0.35);
        let emit_rgb = max(disc.albedo.rgb, vec3(0.15)) * min(disc.emission * 2.2, 4.0);
        let emit_l = dot(emit_rgb, frame.luminance);
        let ndotl = max(dot(normal, ray_dir), 0.05);
        let p_hat = (emit_l * ndotl) / (0.85 + 0.28 * d2);
        let xi = fract(seed + 0.7548777);
        w_sum += p_hat * 2.0;
        candidates += 1.0;
        if xi * max(w_sum, 1e-6) <= p_hat * 2.0 {
            win_pos = disc.position;
            win_col = emit_rgb;
            win_phat = p_hat;
            win_confirmed = true;
        }
    }

    if win_phat <= 1e-4 || candidates <= 0.0 {
        return vec3(0.0);
    }

    let to_win = win_pos - pos;
    let win_dist = length(to_win);
    let win_dir = to_win / max(win_dist, 1e-4);
    var visible = win_confirmed;
    if !visible && win_dist > 0.85 {
        let vis_hit = trace(origin, win_dir, win_dist - 0.75);
        visible = vis_hit.distance < 0.0 || vis_hit.distance >= win_dist - 0.95 || vis_hit.emission > 0.04;
    } else if !visible {
        visible = true;
    }
    if !visible {
        return vec3(0.0);
    }

    let ris_w = clamp(w_sum / max(candidates * win_phat, 1e-4), 0.25, 2.2);
    let win_lum = max(dot(win_col, frame.luminance), 1e-4);
    let unit_col = win_col / win_lum;
    let range_fade = (1.0 - smoothstep(14.0, 24.0, win_dist)) * (1.0 - smoothstep(64.0, 88.0, view_dist));
    return unit_col * (win_phat * ris_w * 1.85 * range_fade);
}

fn compute(input: ComputeInput) {
    let size = vec2<i32>(textureDimensions(gi_history));
    if !input.in_bounds || any(vec2<i32>(input.cell.xy) >= size) {
        return;
    }
    let cell = vec2<i32>(input.cell.xy);
    if frame.dimension != 0 {
        textureStore(gi_history, cell, vec4<u32>(0u));
        return;
    }

    let cur_gi = textureLoad(indirect_gi, cell, 0);
    let cur_ao = textureLoad(contact_visibility, cell, 0);
    let cur_shadow = cur_ao.b;

    let inv_size = 1.0 / vec2<f32>(size);
    let uv = (vec2<f32>(cell) + 0.5) * inv_size;
    let depth = depth_at(uv);
    if depth <= 0.0 {
        textureStore(gi_history, cell, vec4<u32>(0u));
        return;
    }

    let pos = view_position(uv, depth);
    let view_dist = length(pos);
    let full_size = vec2<f32>(textureDimensions(scene_depth));
    let full_cell = clamp(vec2<i32>(uv * full_size), vec2(0), vec2<i32>(full_size) - 1);
    let surf = surface_at(full_cell);
    var di_radiance = vec3(0.0);
    if surf.present && !surface_is_lava(u32(surf.data)) {
        di_radiance = rt_emissive_restir_di(pos, surf.normal, cell, view_dist);
    }

    // 5-tap cross neighborhood statistics for GI variance clamp + SVGF shadow moment estimation.
    let c0 = max(cur_gi.rgb, vec3(0.0)) + di_radiance;
    let cell_l = clamp(cell + vec2(-1, 0), vec2(0), size - 1);
    let cell_r = clamp(cell + vec2(1, 0), vec2(0), size - 1);
    let cell_d = clamp(cell + vec2(0, -1), vec2(0), size - 1);
    let cell_u = clamp(cell + vec2(0, 1), vec2(0), size - 1);

    let cl = max(textureLoad(indirect_gi, cell_l, 0).rgb, vec3(0.0));
    let cr = max(textureLoad(indirect_gi, cell_r, 0).rgb, vec3(0.0));
    let cd = max(textureLoad(indirect_gi, cell_d, 0).rgb, vec3(0.0));
    let cu = max(textureLoad(indirect_gi, cell_u, 0).rgb, vec3(0.0));
    let gi_center = max(cur_gi.rgb, vec3(0.0));
    let mu = (gi_center + cl + cr + cd + cu) * 0.2 + di_radiance;
    let m2 = (gi_center * gi_center + cl * cl + cr * cr + cd * cd + cu * cu) * 0.2;
    let sigma = sqrt(max(m2 - (mu - di_radiance) * (mu - di_radiance), vec3(0.0)));
    let gi_min = max(mu - sigma * 1.35 - di_radiance * 0.35, vec3(0.0));
    let gi_max = mu + sigma * 1.35 + di_radiance * 0.45 + vec3(0.015);

    let sl = textureLoad(contact_visibility, cell_l, 0).b;
    let sr = textureLoad(contact_visibility, cell_r, 0).b;
    let sd = textureLoad(contact_visibility, cell_d, 0).b;
    let su = textureLoad(contact_visibility, cell_u, 0).b;
    let sh_mu = (cur_shadow + sl + sr + sd + su) * 0.2;
    let sh_m2 = (cur_shadow * cur_shadow + sl * sl + sr * sr + sd * sd + su * su) * 0.2;
    let sh_sigma = sqrt(max(sh_m2 - sh_mu * sh_mu, 0.0));

    // AMD FidelityFX-style variance-guided spatial pre-filter for jittered distant shadows:
    // skips uniform lit/shadowed tiles (sh_sigma <= 0.05), preserves sharp terrain edges via depth weights,
    // and down-weights 0.0 vertical terrace risers when cur_shadow > 0.25 so terraced hillsides never stripe.
    var filtered_sh = cur_shadow;
    if view_dist >= 88.0 && sh_sigma > 0.05 {
        let dl = depth_at((vec2<f32>(cell_l) + 0.5) * inv_size);
        let dr = depth_at((vec2<f32>(cell_r) + 0.5) * inv_size);
        let dd = depth_at((vec2<f32>(cell_d) + 0.5) * inv_size);
        let du = depth_at((vec2<f32>(cell_u) + 0.5) * inv_size);
        let tol = max(depth * 0.08, 1e-6);
        let riser_guard = step(0.25, cur_shadow);
        let wl = (1.0 - smoothstep(0.0, tol, abs(dl - depth))) * mix(1.0, step(0.05, sl), riser_guard * 0.85);
        let wr = (1.0 - smoothstep(0.0, tol, abs(dr - depth))) * mix(1.0, step(0.05, sr), riser_guard * 0.85);
        let wd = (1.0 - smoothstep(0.0, tol, abs(dd - depth))) * mix(1.0, step(0.05, sd), riser_guard * 0.85);
        let wu = (1.0 - smoothstep(0.0, tol, abs(du - depth))) * mix(1.0, step(0.05, su), riser_guard * 0.85);
        let w_tot = 1.2 + wl + wr + wd + wu;
        let spatial_avg = (cur_shadow * 1.2 + sl * wl + sr * wr + sd * wd + su * wu) / w_tot;
        let dark_preserve = max(spatial_avg, cur_shadow * 0.90);
        filtered_sh = mix(cur_shadow, dark_preserve, smoothstep(0.05, 0.24, sh_sigma) * 0.70);
    }

    let prev_clip = frame.previous_view_projection * vec4(pos + frame.camera_motion, 1.0);
    let valid_proj = prev_clip.w > 0.05 && frame.delta_time > 0.0 && frame.delta_time < 0.20;
    let prev_ndc = prev_clip.xyz / max(prev_clip.w, 1e-5);
    let prev_uv = 0.5 + prev_ndc.xy * vec2(0.5, -0.5);
    let in_bounds = valid_proj && all(prev_uv > vec2(0.002)) && all(prev_uv < vec2(0.998));

    if !in_bounds {
        textureStore(gi_history, cell, pack_u32_vec4(vec4(pack_hdr8(c0), filtered_sh)));
        return;
    }

    // Manual 2x2 bilinear unpack from r32uint gi_history_previous.
    let p_f = prev_uv * vec2<f32>(size) - 0.5;
    let p0 = clamp(vec2<i32>(floor(p_f)), vec2(0), size - 2);
    let pf = clamp(fract(p_f), vec2(0.0), vec2(1.0));
    let h00 = unpack_u32_vec4(textureLoad(gi_history_previous, p0, 0));
    let h10 = unpack_u32_vec4(textureLoad(gi_history_previous, p0 + vec2(1, 0), 0));
    let h01 = unpack_u32_vec4(textureLoad(gi_history_previous, p0 + vec2(0, 1), 0));
    let h11 = unpack_u32_vec4(textureLoad(gi_history_previous, p0 + vec2(1, 1), 0));
    let hist = mix(mix(h00, h10, pf.x), mix(h01, h11, pf.x), pf.y);
    var eff_hist = unpack_hdr8(hist.rgb);
    var eff_c0 = c0;
    var eff_min = gi_min;
    var eff_max = gi_max;
    if blocklight_mode != 0u && frame.has_ray_tracing != 0u && view_dist < 88.0 {
        // ReSTIR Spatial-Temporal Reuse: gather 4 depth-compatible cross neighbors from previous
        // converged history to dissolve candidate/penumbra noise without smearing wall edges.
        let p_cell = clamp(vec2<i32>(prev_uv * vec2<f32>(size)), vec2(0), size - 1);
        var sp_sum = eff_hist;
        var sp_w = 1.0;
        var sp_max = eff_hist;
        let inv_z = 1.0 / max(depth * 0.06, 1e-6);
        for (var t = 0u; t < 4u; t += 1u) {
            var q_off = vec2<i32>(0);
            if t == 0u { q_off.x = -2; } else if t == 1u { q_off.x = 2; }
            else if t == 2u { q_off.y = -2; } else { q_off.y = 2; }
            let q = clamp(p_cell + q_off, vec2(0), size - 1);
            let q_d = depth_at((vec2<f32>(q) + 0.5) * inv_size);
            let w_d = 1.0 - smoothstep(0.4, 1.8, abs(q_d - depth) * inv_z);
            let q_rgb = unpack_hdr8(unpack_u32_vec4(textureLoad(gi_history_previous, q, 0)).rgb);
            sp_sum += q_rgb * w_d;
            sp_w += w_d;
            sp_max = max(sp_max, q_rgb * w_d);
        }
        let sp_avg = sp_sum / max(sp_w, 1e-4);
        eff_hist = mix(eff_hist, sp_avg, 0.55);
        eff_c0 = mix(c0, sp_avg + 0.45 * (di_radiance - max(sp_avg - gi_center, vec3(0.0))), 0.35);
        eff_min = max(min(gi_min, sp_avg * 0.55), vec3(0.0));
        eff_max = max(gi_max, sp_max + di_radiance * 0.65);
    }
    let clamped_gi = clamp(eff_hist, eff_min, eff_max);
    // SVGF moment-guided shadow clamp: keep upper clamp generous when cur_shadow is active
    // so alternating 0.0 vertical terrace risers never drag temporal shadow history down.
    let sh_min = max(max(sh_mu - sh_sigma * 1.35 - 0.02, cur_shadow * 0.78), 0.0);
    let sh_max = min(max(sh_mu + sh_sigma * 1.35 + 0.02, cur_shadow), 1.0);
    let clamped_sh = clamp(hist.a, sh_min, sh_max);

    let motion_px = length((uv - prev_uv) * vec2<f32>(size));
    let gi_alpha = mix(0.14, 0.42, clamp(motion_px * 0.18, 0.0, 1.0));
    // Keep near-field (<24m) contact shadows instant/razor-sharp; accumulate distant shadows smoothly.
    let sh_alpha = select(1.0, mix(0.16, 0.45, clamp(motion_px * 0.16, 0.0, 1.0)), view_dist >= 24.0);

    let out_gi = mix(clamped_gi, eff_c0, gi_alpha);
    let out_sh = mix(clamped_sh, filtered_sh, sh_alpha);
    textureStore(gi_history, cell, pack_u32_vec4(vec4(pack_hdr8(out_gi), out_sh)));
}
